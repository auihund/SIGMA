export WANDB_API_KEY=$YOUR_KEY


torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port=12344 \
  train/finetune_specialtoken.py \
  --num_shard 4 \
  --use_lora False \
  --visual_gen True \
  --visual_und False \
  --save_every 5000 \
  --total_steps 30000 \
  --log_every 1 \
  --warmup_steps 0 \
  --lr 2e-5 \
  --lr_scheduler cosine \
  --min_lr 1e-7 \
  --dataset_config_file ./data/configs/specialtoken.yaml \
  --model_path ./models/BAGEL-7B-MoT \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from ./models/BAGEL-7B-MoT \
  --finetune_from_hf True \
  --use_mask_attn False \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --num_worker 1 \
  --expected_num_tokens 30000 \
  --max_num_tokens 30000 \
  --max_num_tokens_per_sample 30000 \
  --results_dir results/specialtoken \
  --checkpoint_dir results/specialtoken/checkpoints \
  --wandb_project SIGMA \
  --wandb_name results_specialtoken \
  --wandb_runid 0 \
  --wandb_offline False

  # --lr 2e-5 \
#   --resume-from ./BAGEL-7B-MoT \
#   --auto_resume True \
#   --resume-model-only True \
  # --log_every 1 \
