master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Get the filename without extension
export WANDB_MODE=disabled

filename=$(basename "$0" | cut -f 1 -d '.')

dir_path=PointLLM_finalloss

model_name_or_path= # Path to the output dir of stage 1 training
data_path=/ENEL/data/objaverse_data
anno_path=/ENEL/data/anno_data/PointLLM_complex_instruction_70K.json
output_dir=/ENEL/outputs/Train_stage2/$filename

cmd="PYTHONPATH=$dir_path:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nnodes=1 --nproc_per_node=4 --master_port=$master_port pointllm/train/train_mem.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --anno_path $anno_path \
    --output_dir $output_dir \
    --version v1 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --fix_llm False \
    --fix_pointnet False \
    --report_to wandb \
    --run_name $filename \
    --gradient_checkpointing True \
    --stage_2 True \
    --fsdp \"full_shard auto_wrap\" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --conversation_types "detailed_description" "single_round" "multi_round" \
    --mm_projector_lr 2e-5 \
    --vision_tower_lr 2e-5 \
    --group_size 81 \
    --num_stages 3 \
    --embed_dim 288 \
    --LGA_dim 2 2 2 \
    --input_points 1024 \
    --tune_layer 0 \
    --mask_ratio 0.1 \
    --recon_fp 0 \
    --mae_fp 1 \
    --mask_dim 4096 \
    --mae_feature 0 \
    --recon_feature 0 \
    --recon_pos 1 \
    --pos_embed_mae 0 \
    --pos_embed_dim 4096 \
    --use_color True" 

eval "$cmd"
