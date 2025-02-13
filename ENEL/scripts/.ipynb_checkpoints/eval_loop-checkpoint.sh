#!/bin/bash

# Configuration variables
export PYTHONPATH="/mnt/petrelfs/tangyiwen/PointLLM_finalloss:$PYTHONPATH"
MODEL_NAMES=(
"/mnt/petrelfs/tangyiwen/PointLLM_finalloss/outputs/PointLLM_train_stage2_mae03_32_featuremse_after_recon32_patch_t_posall_4_1286432_168_res1_max_bottle512_trans_mid2/job_script_1736912760"
    )
LOG_SUFFIXES=(
    "PointLLM_train_stage2_mae03_32_featuremse_after_recon32_patch_t_posall_4_1286432_168_res1_max_bottle512_trans_mid2"
)
EVAL_DIR_SUFFIX="evaluation"

# Function to check if folder exists and is not empty
check_evaluation_ready() {
    local model_name=$1
    eval_dir="${model_name}/${EVAL_DIR_SUFFIX}"
    if [ -d "$eval_dir" ] && [ "$(ls -A $eval_dir)" ]; then
        return 0
    else
        return 1
    fi
}

# Run evaluation script for each model
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    LOG_SUFFIX="${LOG_SUFFIXES[$i]}"

    echo "Processing MODEL_NAME: $MODEL_NAME with LOG_SUFFIX: $LOG_SUFFIX"

    # Replace MODEL_NAME and LOG_SUFFIX in eval.sh and execute
    bash /mnt/petrelfs/tangyiwen/PointLLM_finalloss/scripts/eval.sh "$MODEL_NAME" "$LOG_SUFFIX"

    # Wait until evaluation directory is ready
    echo "Waiting for ${MODEL_NAME}/${EVAL_DIR_SUFFIX} to contain files..."
    until check_evaluation_ready "$MODEL_NAME"; do
        sleep 60
    done
    echo "${MODEL_NAME}/${EVAL_DIR_SUFFIX} is ready."

    # If 4 models have been processed, validate the first one
    if [ $((i + 1)) -eq 4 ]; then
        echo "Validating files in ${MODEL_NAMES[0]}/${EVAL_DIR_SUFFIX}..."
        if [ "$(ls -A ${MODEL_NAMES[0]}/${EVAL_DIR_SUFFIX} | wc -l)" -eq 4 ]; then
            echo "Validation successful: Found 4 files in ${MODEL_NAMES[0]}/${EVAL_DIR_SUFFIX}."
        else
            echo "Validation failed: Not all files are present in ${MODEL_NAMES[0]}/${EVAL_DIR_SUFFIX}."
            echo "Rechecking in 60 seconds..."
            while [ "$(ls -A ${MODEL_NAMES[0]}/${EVAL_DIR_SUFFIX} | wc -l)" -ne 4 ]; do
                sleep 90
                echo "Rechecking..."
            done
            echo "Validation successful: Found 4 files in ${MODEL_NAMES[0]}/${EVAL_DIR_SUFFIX}."
        fi
    fi
done

echo "All evaluations completed successfully."
