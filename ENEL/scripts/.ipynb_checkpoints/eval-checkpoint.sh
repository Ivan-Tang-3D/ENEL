# #Objaveerse_Captioning
# srun -p optimal --quotatype=reserved --gres=gpu:1 -J leo python pointllm/eval/eval_objaverse.py --model_name /mnt/petrelfs/tangyiwen/PointLLM/outputs/PointLLM_train_stage2_nopos_lrsame_sample165K_PointPN_3stage_complete_learninit_group32_dim192/job_script_1729218197 --task_type captioning --prompt_index 2 > /mnt/petrelfs/tangyiwen/PointLLM/logs/try_obj_nopos_lrsame_sample165K_PointPN_3stage_complete_learninit_group32_dim192.log 2>&1 &

# # Open Vocabulary Classification on Objaverse
# srun -p optimal --quotatype=reserved --gres=gpu:1 -J leo python pointllm/eval/eval_objaverse.py --model_name /mnt/petrelfs/tangyiwen/PointLLM/outputs/PointLLM_train_stage2_nopos_lrsame_sample165K_PointPN_3stage_complete_learninit_group32_dim192/job_script_1729218197 --task_type classification --prompt_index 0 > /mnt/petrelfs/tangyiwen/PointLLM/logs/try_objcls_nopos_lrsame_sample165K_PointPN_3stage_complete_learninit_group32_dim192.log 2>&1 & # or --prompt_index 1

# # Close-set Zero-shot Classification on ModelNet40
# srun -p optimal --quotatype=reserved --gres=gpu:1 -J leo python pointllm/eval/eval_modelnet_cls.py --model_name /mnt/petrelfs/tangyiwen/PointLLM/outputs/PointLLM_train_stage2_nopos_lrsame_sample165K_PointPN_3stage_complete_learninit_group32_dim192/job_script_1729218197 --prompt_index 0 > /mnt/petrelfs/tangyiwen/PointLLM/logs/try_objcls_modelnet_nopos_lrsame_sample165K_PointPN_3stage_complete_learninit_group32_dim192.log 2>&1 # or --prompt_index 1

# # Open Vocabulary Classification on Objaverse
# srun -p optimal --quotatype=reserved --gres=gpu:1 -J leo python pointllm/eval/evaluator.py --results_path /mnt/petrelfs/tangyiwen/PointLLM/outputs/PointLLM_train_stage2_nopos_lrsame_sample165K_PointPN_3stage_complete_learninit_group64_dim96/job_script_1729219230/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt0.json --model_type gpt-4-0613 --eval_type open-free-form-classification > /mnt/petrelfs/tangyiwen/PointLLM/logs/try_objcls_caption_gpt_nopos_lrsame_sample165K_PointPN_3stage_complete_learninit_group64_dim96.log 2>&1 & #--parallel --num_workers 15

# #Object captioning on Objaverse
# srun -p optimal --quotatype=reserved --gres=gpu:1 -J leo python pointllm/eval/evaluator.py --results_path /mnt/petrelfs/tangyiwen/PointLLM/outputs/PointLLM_train_stage2_nopos_lrsame_sample165K_PointPN_3stage_complete_learninit_group64_dim192/job_script_1729219474/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json --model_type gpt-4-0613 --eval_type object-captioning > /mnt/petrelfs/tangyiwen/PointLLM/logs/try_obj_caption_gpt_nopos_lrsame_sample165K_PointPN_3stage_complete_learninit_group64_dim192.log 2>&1 &

# #Close-set Zero-shot Classification on ModelNet40
# srun -p optimal --quotatype=reserved --gres=gpu:1 -J leo python pointllm/eval/evaluator.py --results_path /mnt/petrelfs/tangyiwen/PointLLM/outputs/PointLLM_train_stage2_nopos_lrsame_sample165K_PointPN_3stage_complete_learninit_group64_dim192/job_script_1729219474/evaluation/ModelNet_classification_prompt0.json --model_type gpt-3.5-turbo --eval_type modelnet-close-set-classification  > /mnt/petrelfs/tangyiwen/PointLLM/logs/try_objcls_modelnet_gpt_nopos_lrsame_sample165K_PointPN_3stage_complete_learninit_group64_dim192.log 2>&1 & #--parallel --num_workers 15

# #Object captioning on Objaverse trans
# srun -p optimal --quotatype=reserved --gres=gpu:1 -J leo python pointllm/eval/traditional_evaluator.py --results_path /mnt/petrelfs/tangyiwen/PointLLM/outputs/PointLLM_train_stage2_nopos_lrsame_sample165K_PointPN_3stage_complete_learninit_group64_dim96/job_script_1729219230/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json > /mnt/petrelfs/tangyiwen/PointLLM/logs/try_obj_caption_trans_nopos_lrsame_sample165K_PointPN_3stage_complete_learninit_group64_dim96.log 2>&1 &
PARTITION="optimal"
QUOTATYPE="spot"
GRES="gpu:1"
JOB_NAME="leo"
# MODEL_NAME="$1"
# LOG_SUFFIX="$2"
MODEL_NAME="/mnt/petrelfs/tangyiwen/PointLLM_finalloss/outputs/PointLLM_train_stage2_mae03_32_featuremse_after_recon32_patch_t_posall_4_1286432_168_maxmean_mid4_poolingnice_coef07_attn256_gate_nores_inter4/job_script_1739107264"
LOG_SUFFIX="try_obj_mae03_32_featuremse_after_recon32_patch_t_posall_4_1286432_168_maxmean_mid4_poolingnice_coef07_attn256_gate_nores_inter4"
LOG_DIR="/mnt/petrelfs/tangyiwen/PointLLM_finalloss/new_eval_logs_2"
LOG_EDIR="/mnt/petrelfs/tangyiwen/PointLLM_finalloss/new_eval_logs_2"

unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
export http_proxy=http://tangyiwen:nV5E1LMFeYZEk1HdUpzUGcrBvEqAIpaAxWAcEf0hGc454TxBCDukIiERvt54@10.1.20.50:23128/ ; export https_proxy=http://tangyiwen:nV5E1LMFeYZEk1HdUpzUGcrBvEqAIpaAxWAcEf0hGc454TxBCDukIiERvt54@10.1.20.50:23128/ ; export HTTP_PROXY=http://tangyiwen:nV5E1LMFeYZEk1HdUpzUGcrBvEqAIpaAxWAcEf0hGc454TxBCDukIiERvt54@10.1.20.50:23128/ ; export HTTPS_PROXY=http://tangyiwen:nV5E1LMFeYZEk1HdUpzUGcrBvEqAIpaAxWAcEf0hGc454TxBCDukIiERvt54@10.1.20.50:23128/
# 定义命令列表
# commands=(
#     "python pointllm/eval/eval_objaverse.py --model_name $MODEL_NAME --task_type captioning --prompt_index 2 > $LOG_DIR/try_obj_${LOG_SUFFIX}.log 2>&1 &"
#     "python pointllm/eval/eval_objaverse.py --model_name $MODEL_NAME --task_type classification --prompt_index 0 > $LOG_DIR/try_objcls_${LOG_SUFFIX}.log 2>&1 &"
#     "python pointllm/eval/eval_modelnet_cls.py --model_name $MODEL_NAME --prompt_index 0 > $LOG_DIR/try_objcls_modelnet_${LOG_SUFFIX}.log 2>&1 &"
# )
#CUDA_VISIBLE_DEVICES=0,1 
#srun -p $PARTITION --quotatype=$QUOTATYPE --gres=$GRES -J $JOB_NAME 
CUDA_VISIBLE_DEVICES=4 python pointllm/eval/eval_objaverse.py --model_name $MODEL_NAME --task_type captioning --prompt_index 2 #> $LOG_EDIR/try_obj_${LOG_SUFFIX}.log 2>&1 &
# #srun -p $PARTITION --quotatype=$QUOTATYPE --gres=$GRES -J $JOB_NAME 
# CUDA_VISIBLE_DEVICES=6 python pointllm/eval/eval_objaverse.py  --model_name $MODEL_NAME --task_type classification --prompt_index 0 > $LOG_EDIR/try_objcls_${LOG_SUFFIX}.log 2>&1 &
#srun -p $PARTITION --quotatype=$QUOTATYPE --gres=$GRES -J $JOB_NAME python pointllm/eval/eval_modelnet_cls.py --group_size 32 --num_stages 3 --embed_dim 240 --LGA_dim "123" --model_name $MODEL_NAME --prompt_index 0 > $LOG_DIR/try_objcls_modelnet_${LOG_SUFFIX}.log 2>&1 &
#CUDA_VISIBLE_DEVICES=4,5 python pointllm/eval/eval_modelnet_cls.py --group_size 32 --num_stages 3 --embed_dim 240 --LGA_dim "123" --model_name $MODEL_NAME --prompt_index 0 > $LOG_DIR/try_objcls_modelnet_${LOG_SUFFIX}.log 2>&1 &

# 执行前三个命令
# for cmd in "${commands[@]}"; do
#     srun -p $PARTITION --quotatype=$QUOTATYPE --gres=$GRES -J $JOB_NAME $cmd &
# done

# 等待前三个命令完成
# waits

# 定义后四个命令及其对应的results_path
CLS_OBJAVERSE="${MODEL_NAME}/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt0.json"
CAPTION_OBJAVERSE="${MODEL_NAME}/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json"
CLS_MODEL="${MODEL_NAME}/evaluation/ModelNet_classification_prompt0.json"

# declare -A cmd_results_paths=(
#     ["python pointllm/eval/evaluator.py --results_path $CLS_OBJAVERSE --model_type gpt-4-0613 --eval_type open-free-form-classification > $LOG_DIR/try_objcls_caption_gpt_${LOG_SUFFIX}.log 2>&1 &"]=$CLS_OBJAVERSE
#     ["python pointllm/eval/evaluator.py --results_path $CAPTION_OBJAVERSE --model_type gpt-4-0613 --eval_type object-captioning > $LOG_DIR/try_obj_caption_gpt_${LOG_SUFFIX}.log 2>&1 &"]=$CAPTION_OBJAVERSE
#     ["python pointllm/eval/traditional_evaluator.py --results_path $CAPTION_OBJAVERSE > $LOG_DIR/try_obj_caption_trans_${LOG_SUFFIX}.log 2>&1 &"]=$CAPTION_OBJAVERSE
#     ["python pointllm/eval/evaluator.py --results_path $CLS_MODEL --model_type gpt-3.5-turbo --eval_type modelnet-close-set-classification > $LOG_DIR/try_objcls_modelnet_gpt_${LOG_SUFFIX}.log 2>&1 &"]=$CLS_MODEL
# )

# seen_results_paths=()

# while [ ! -f "$CLS_OBJAVERSE" ]; do
#     echo "等待 $CLS_OBJAVERSE 文件生成..."
#     sleep 60  # 每60秒检查一次
# done
# unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
# export http_proxy=http://closeai-proxy.pjlab.org.cn:23128 ; export https_proxy=http://closeai-proxy.pjlab.org.cn:23128 ; export HTTP_PROXY=http://closeai-proxy.pjlab.org.cn:23128 ; export HTTPS_PROXY=http://closeai-proxy.pjlab.org.cn:23128

# echo "$CLS_OBJAVERSE 文件已生成，开始执行命令"
# #srun -p $PARTITION --quotatype=$QUOTATYPE --gres=$GRES -J $JOB_NAME 
# python pointllm/eval/evaluator.py --results_path $CLS_OBJAVERSE --model_type gpt-4-0613 --eval_type open-free-form-classification > $LOG_EDIR/try_objcls_caption_gpt_${LOG_SUFFIX}.log 2>&1 &

# while [ ! -f "$CAPTION_OBJAVERSE" ]; do
#     echo "等待 $CAPTION_OBJAVERSE 文件生成..."
#     sleep 60  # 每60秒检查一次
# done
# echo "$CAPTION_OBJAVERSE 文件已生成，开始执行命令"
# #srun -p $PARTITION --quotatype=$QUOTATYPE --gres=$GRES -J $JOB_NAME 
# python pointllm/eval/evaluator.py --results_path $CAPTION_OBJAVERSE --model_type gpt-4-0613 --eval_type object-captioning > $LOG_EDIR/try_obj_caption_gpt_${LOG_SUFFIX}.log 2>&1 &

# # while [ ! -f "$CLS_MODEL" ]; do
# #     echo "等待 $CLS_MODEL 文件生成..."
# #     sleep 60  # 每60秒检查一次
# # done
# # echo "$CLS_MODEL 文件已生成，开始执行命令"
# # #srun -p $PARTITION --quotatype=$QUOTATYPE --gres=$GRES -J $JOB_NAME 
# # python pointllm/eval/evaluator.py --results_path $CLS_MODEL --model_type gpt-3.5-turbo --eval_type modelnet-close-set-classification #> $LOG_EDIR/try_objcls_modelnet_gpt_${LOG_SUFFIX}.log 2>&1 &

# while [ ! -f "$CAPTION_OBJAVERSE" ]; do
#     echo "等待 $CAPTION_OBJAVERSE 文件生成..."
#     sleep 60  # 每60秒检查一次
# done
# # unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
# # export http_proxy=http://tangyiwen:nV5E1LMFeYZEk1HdUpzUGcrBvEqAIpaAxWAcEf0hGc454TxBCDukIiERvt54@10.1.20.50:23128/ ; export https_proxy=http://tangyiwen:nV5E1LMFeYZEk1HdUpzUGcrBvEqAIpaAxWAcEf0hGc454TxBCDukIiERvt54@10.1.20.50:23128/ ; export HTTP_PROXY=http://tangyiwen:nV5E1LMFeYZEk1HdUpzUGcrBvEqAIpaAxWAcEf0hGc454TxBCDukIiERvt54@10.1.20.50:23128/ ; export HTTPS_PROXY=http://tangyiwen:nV5E1LMFeYZEk1HdUpzUGcrBvEqAIpaAxWAcEf0hGc454TxBCDukIiERvt54@10.1.20.50:23128/

# echo "$CAPTION_OBJAVERSE 文件已生成，开始执行命令"
# #srun -p $PARTITION --quotatype=$QUOTATYPE --gres=$GRES -J $JOB_NAME 
# CUDA_VISIBLE_DEVICES=5 python pointllm/eval/traditional_evaluator.py --results_path $CAPTION_OBJAVERSE > $LOG_EDIR/try_obj_caption_trans_${LOG_SUFFIX}.log 2>&1 &

# 执行后四个命令，检查results_path是否存在
# for cmd in "${!cmd_results_paths[@]}"; do
#     results_path="${cmd_results_paths[$cmd]}"
#     while [ ! -f "$results_path" ]; do
#         echo "等待 $results_path 文件生成..."
#         sleep 60  # 每60秒检查一次
#     done
#         # 检查results_path是否已经出现过
#     if [[ " ${seen_results_paths[@]} " =~ " ${results_path} " ]]; then
#         unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
#         export http_proxy=http://tangyiwen:nV5E1LMFeYZEk1HdUpzUGcrBvEqAIpaAxWAcEf0hGc454TxBCDukIiERvt54@10.1.20.50:23128/ ; export https_proxy=http://tangyiwen:nV5E1LMFeYZEk1HdUpzUGcrBvEqAIpaAxWAcEf0hGc454TxBCDukIiERvt54@10.1.20.50:23128/ ; export HTTP_PROXY=http://tangyiwen:nV5E1LMFeYZEk1HdUpzUGcrBvEqAIpaAxWAcEf0hGc454TxBCDukIiERvt54@10.1.20.50:23128/ ; export HTTPS_PROXY=http://tangyiwen:nV5E1LMFeYZEk1HdUpzUGcrBvEqAIpaAxWAcEf0hGc454TxBCDukIiERvt54@10.1.20.50:23128/
#     else
#         unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
#         export http_proxy=http://closeai-proxy.pjlab.org.cn:23128 ; export https_proxy=http://closeai-proxy.pjlab.org.cn:23128 ; export HTTP_PROXY=http://closeai-proxy.pjlab.org.cn:23128 ; export HTTPS_PROXY=http://closeai-proxy.pjlab.org.cn:23128
#     fi
#     echo "$results_path 文件已生成，开始执行命令"
#     seen_results_paths+=("$results_path")
#     srun -p $PARTITION --quotatype=$QUOTATYPE --gres=$GRES -J $JOB_NAME $cmd
# done