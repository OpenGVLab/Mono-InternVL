#!/bin/bash

GPUS=8                         
PER_DEVICE_BATCH_SIZE=4         
BATCH_SIZE=128                  
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS)) 

MODEL=${MODEL:-"Path to your model"}
OUTPUT_DIR=${OUTPUT_DIR:-"Path to your output directory"}
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun --nproc_per_node=$GPUS --master_port=29501 \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path ${MODEL} \
  --vision_type patch \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data_llava_finetune.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --pad2square False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --unfreeze_ve True \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 3000 \
  --save_total_limit 3 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 2048 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "./shell/zero_stage1_config.json" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
