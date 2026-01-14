# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import math
import random
from PIL import Image

import torch
from torch.nn.attention.flex_attention import or_masks, and_masks


def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return (~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx])))

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ['full', 'noise'] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == 'noise' else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)

    document_id = torch.cat([torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]).to(device)

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


def get_flattened_position_ids_interpolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    boundaries = torch.arange(1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w).flatten()
    return pos_ids


def prepare_attention_mask_per_sample(split_lens, attn_modes, device="cpu"):
    """
    nested_split_lens: A list of N lists of ints. Each int indicates the length of a split within 
        a sample, where each sample contains multiple splits with different attn modes.
    nested_attn_modes: whether to use full attn in each split.
    """
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        assert attn_mode in ['causal', 'full', 'noise']
        if attn_mode == "causal":
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s), device=device).tril()
            attention_mask[csum:csum + s, :csum] = 1
        else:
            attention_mask[csum:csum + s, csum:csum + s] = torch.ones((s, s))
            attention_mask[csum:csum + s, :csum] = 1
        csum += s

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "noise":
            attention_mask[:, csum : csum + s] = torch.zeros((sample_len, s))
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s))
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float).masked_fill_(
        ~attention_mask, float("-inf")
    )

    return attention_mask





def apply_triplet_local_bias(extra_bias, A_span, C_span, X_span, minus_inf=-1e4):
    """
    组内规则（改后）：
      - C 行：允许看 A、C；仅屏蔽 C→X
      - X 行：允许看 A、C；屏蔽 X→X
      - A 行：允许看 A、C、X（保持不变）
    组间隔离继续由 _apply_group_isolation 统一处理。
    """
    a0,a1 = A_span; c0,c1 = C_span; x0,x1 = X_span
    # C 行：只屏蔽 X；不再屏蔽 C→C
    extra_bias[c0:c1, x0:x1] = minus_inf
    # # X 行：只屏蔽 X 自身（X→X）
    # extra_bias[x0:x1, x0:x1] = minus_inf
    # A 行：不屏蔽组内任何列
    return extra_bias






def apply_x_block_other_groups_c(extra_bias, groups, minus_inf=float("-inf")):
    """
    屏蔽：每组的 X 行 -> 其它组的 C 列。保留本组 C 列可见。
    groups: [{'A':(a0,a1), 'C':(c0,c1), 'X':(x0,x1)}, ...]
    extra_bias: [T, T]
    """
    T = extra_bias.size(-1)

    def span_mask(span):
        m = torch.zeros(T, dtype=torch.bool, device=extra_bias.device)
        s, e = span
        if e > s:
            m[s:e] = True
        return m

    # dtype 安全的 -inf（避免半精度下的 -inf 数值问题）
    if not torch.isfinite(torch.tensor(minus_inf)):
        # 用一个大负数代替 -inf
        minus_inf_val = -1e9 if extra_bias.dtype in (torch.float16, torch.bfloat16) else -1e30
    else:
        minus_inf_val = minus_inf
    minus_inf_val = torch.tensor(minus_inf_val, dtype=extra_bias.dtype, device=extra_bias.device)

    # 预先收集所有组的 C 掩码
    group_C_cols = [span_mask(g["C"]) for g in groups]

    for gi, g in enumerate(groups):
        rows_X = span_mask(g["X"])
        if not rows_X.any():
            continue

        # 其它组的 C 列并集
        others_C = torch.zeros(T, dtype=torch.bool, device=extra_bias.device)
        for gj, Cmask in enumerate(group_C_cols):
            if gj == gi:
                continue
            others_C |= Cmask

        # 关键修复：显式移除“本组的 C 列”，
        # 即便其它组的 C 与本组相同，也不会把自己的 C 列屏蔽掉
        if group_C_cols[gi].any():
            others_C = others_C & (~group_C_cols[gi])

        if others_C.any():
            # 取出 X 行 → 改列 → 写回，避免链式索引拷贝
            tmp = extra_bias[rows_X]
            tmp[:, others_C] = minus_inf_val
            extra_bias[rows_X] = tmp

    return extra_bias




def split_integer_exp_decay(S, ng_sample_decay=1.0):
    if ng_sample_decay == 1.0:
        N = random.randint(1, S)
    else:
        base = (1 - ng_sample_decay) / (1 - math.pow(ng_sample_decay, S))
        p = [base * math.pow(ng_sample_decay, i) for i in range(S)]
        N = random.choices(list(range(1, S + 1)), p, k=1)[0]
    cumsum = [0] + sorted(random.sample(range(1, S), N - 1)) + [S]
    result = [cumsum[i+1] - cumsum[i] for i in range(len(cumsum) - 1)]
    return result, cumsum


def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if '<|im_start|>' not in all_special_tokens:
        new_tokens.append('<|im_start|>')

    if '<|im_end|>' not in all_special_tokens:
        new_tokens.append('<|im_end|>')

    if '<|vision_start|>' not in all_special_tokens:
        new_tokens.append('<|vision_start|>')

    if '<|vision_end|>' not in all_special_tokens:
        new_tokens.append('<|vision_end|>')
        



    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    
    

    new_token_ids = dict(
        bos_token_id=bos_token_id, 
        eos_token_id=eos_token_id, 
        start_of_image=start_of_image, 
        end_of_image=end_of_image
    )

    return tokenizer, new_token_ids, num_new_tokens



def len2weight(x, loss_reduction='square'):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)
