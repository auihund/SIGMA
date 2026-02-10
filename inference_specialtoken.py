import os
import pandas as pd
import re
import argparse
import random
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from pathlib import Path
import shutil
from copy import deepcopy
from typing import Any, AsyncIterable, Callable, Dict, Generator, List, NamedTuple, Optional, Tuple, Union
import requests
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from data.transforms import ImageTransform
# from data.data_utils import pil_img2rgb, add_special_tokens
from data.data_utils_token import add_special_tokens
from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
from inferencer import InterleaveInferencer  



def parse_args():
    parser = argparse.ArgumentParser(description='BAGEL Model Inference')
    parser.add_argument('--parquet_dir', type=str, default="./data/all_specialtoken_data_select/seedxedit_multi/test", help='Path to the input Parquet file')
    parser.add_argument('--start_id', type=int, default=0, help='Starting sample ID (default: 0)')
    parser.add_argument('--model_path', type=str, default="./BAGEL-7B-MoT", help='Path to the model directory')
    parser.add_argument('--ema_model_path', type=str, default="./results/all_special_select_lr00002_interleave_specialtoken_cosine_nolora_new/checkpoints/0040000", help='Path to EMA model checkpoint')
    parser.add_argument('--output_dir', type=str, default="./results/all_special_select_lr00002_interleave_specialtoken_cosine_nolora_new/checkpoints/0040000/inference", help='Output directory for generated images')
    parser.add_argument('--max_mem_per_gpu', type=str, default="70GiB", help='Maximum memory per GPU')
    parser.add_argument('--use_lora', type=bool, default=False, help='Whether to use LoRA')
    parser.add_argument('--visual_gen', type=bool, default=True, help='Enable visual generation')
    parser.add_argument('--visual_und', type=bool, default=False, help='Enable visual understanding')
    return parser.parse_args()


def _extract_instruction(instr_field) -> str:
    if instr_field is None:
        return ""
    try:
        if isinstance(instr_field, (list, tuple)) and len(instr_field) > 0:
            first = instr_field[0]
            if isinstance(first, (list, tuple)) and len(first) > 0:
                return str(first[0])
            return str(first)
        # numpy/pyarrow list-like
        return str(instr_field[0][0])
    except Exception:
        try:
            return str(instr_field[0])
        except Exception:
            return str(instr_field)

def _extract_image_list(img_field) -> list:
    if img_field is None:
        return []
    if isinstance(img_field, list):
        return img_field
    if isinstance(img_field, tuple):
        return list(img_field)
    try:
        return list(img_field)  # numpy/pyarrow list-like
    except Exception:
        return []

def save_text_meta(txt_path: Path, instruction: str, data_prep_method: str, parquet_path: Path, sample_id: int):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"parquet_file: {parquet_path}\n")
        f.write(f"sample_id: {sample_id}\n")
        f.write(f"data_prep_method: {data_prep_method}\n")
        f.write("instruction:\n")
        f.write(instruction.strip() + "\n")


def initialize_model_and_inferencer(model_path, ema_model_path, max_mem_per_gpu, use_lora, visual_und, visual_gen):
    os.makedirs(args.output_dir, exist_ok=True)
    # LLM config preparing
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    # Tokenizer Preparing
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    # llm_config.vocab_size = len(tokenizer)
    
    # ViT config preparing
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    # Bagel config preparing
    config = BagelConfig(
        visual_gen=visual_gen,
        visual_und=visual_und,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    
    print(f"use_lora: {use_lora}")
    print(f"visual_und: {visual_und}")
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        if visual_und:
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules="all-linear",
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
    
    # Image Transform Preparing
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)
    # Device mapping
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    if use_lora:
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(ema_model_path, "model.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload"
        )
    else:
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(ema_model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload"
        )
    in_emb = model.language_model.get_input_embeddings().weight.shape[0]
    out_emb = model.language_model.get_output_embeddings().weight.shape[0]
    print("len(tokenizer) =", len(tokenizer), "in_emb =", in_emb, "out_emb =", out_emb)

    # assert len(tokenizer) == in_emb == out_emb, "词表与嵌入层/输出头大小不一致！"
    model = model.eval()
    print('Model loaded')
  
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids
    )
    return inferencer


def generate_image_from_parquet_sample(sample_id, row, inferencer, output_dir):
    inference_hyper = dict(
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )

    instruction = row['instruction_list'][0][0]
    data_prep_method = row.get('data_prep_method', 'unknown')
    txt_path = file_out_dir / f"sample_{global_sample_id:06d}.txt"
    save_text_meta(txt_path, instruction, data_prep_method, parquet_path, global_sample_id)
    image_list = row['image_list']
    input_list = []
    input_list.append(instruction)
    
    for img_idx, img_bytes in enumerate(image_list[:-1]):
        img = Image.open(BytesIO(img_bytes))
        
        img_path = file_out_dir / f"sample_{global_sample_id:06d}_in_{img_idx}.jpg"
        img.save(img_path, quality=95)
        print(f"Saved input image: {img_path}")
        
        input_list.append(img)

    
    output_dict = inferencer.interleave_inference(input_lists=input_list, **inference_hyper)
    generated_img = output_dict[0]
    
    gen_path = file_out_dir / f"sample_{global_sample_id:06d}_gen.jpg"
    generated_img.save(gen_path, quality=95)
    print(f"Saved generated image: {gen_path}")
    
    if len(image_list) > 1:
        # gt_img = Image.open(BytesIO(image_list[-1]))
        # gt_path = file_out_dir / f"sample_{global_sample_id:06d}_img_gt.jpg"
        # gt_img.save(gt_path, quality=95)
        # print(f"Saved ground truth: {gt_path}")
        try:
            gt_img = Image.open(BytesIO(image_list[-1]))
            gt_path = os.path.join(output_dir, f"test_sample{sample_id}_img_gt.jpg")
            gt_img.save(gt_path)
            print(f"Saved ground truth: {gt_path}")
        except Exception as e:
            print(f"[Warning] Failed to load or save ground truth for sample {sample_id}: {e}")
            pass

    print(f"[OK] {parquet_path.name} | sample #{global_sample_id} -> {gen_path.name} + meta/txt")






global_sample_id = 0
file_out_dir = None
parquet_path = None

def main(parquet_dir, start_id, output_dir, inferencer):
    root = Path(parquet_dir)
    files = sorted([p for p in root.glob("*.parquet") if p.is_file()])
    if not files:
        print(f"[WARN] No parquet found in: {parquet_dir}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    global global_sample_id, file_out_dir, parquet_path

    for f in files:
        parquet_path = f 
        subdir = f.stem
        file_out_dir = Path(output_dir) / subdir
        file_out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(parquet_path)  

        for idx, row in enumerate(df.itertuples(index=False)):
            global global_sample_id

            if global_sample_id < start_id:
                if (global_sample_id % 100) == 0:
                    print(f"Skip Sample {global_sample_id}")
                global_sample_id += 1
                continue

          
            instr_field = getattr(row, 'instruction_list')
            img_field   = getattr(row, 'image_list')
            try:
                _instr = _extract_instruction(instr_field)
                _imgs  = _extract_image_list(img_field)
                print(f"Instruction: {_instr}")
                print(f"Include {len(_imgs)} images")


                generate_image_from_parquet_sample(
                    global_sample_id,                                
                    {'instruction_list': instr_field, 'image_list': img_field},
                    inferencer,
                    output_dir
                )
                print(f"Sample {global_sample_id} completed")
            except Exception as e:
                print(f"[ERR] Process {f.name} | sample #{global_sample_id} error: {e}")
            finally:
                global_sample_id += 1                                  



if __name__ == "__main__":
    args = parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    inferencer = initialize_model_and_inferencer(
        model_path=args.model_path,
        ema_model_path=args.ema_model_path,
        max_mem_per_gpu=args.max_mem_per_gpu,
        use_lora=args.use_lora,
        visual_und=args.visual_und,
        visual_gen=args.visual_gen
    )


    main(
        parquet_dir=args.parquet_dir,
        start_id=args.start_id,
        output_dir=args.output_dir,
        inferencer=inferencer
    )
