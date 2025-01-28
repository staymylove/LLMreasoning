#!/bin/bash

# Set the model and other parameters
MODEL_NAME="/root/DeepSeek-R1-Distill-Qwen-14B"
MODEL_TYPE="vllm"  # Set to "vllm" for vLLM or "hf" for Hugging Face
DEVICE="cuda"
MAX_NEW_TOKENS=2048
TEMPERATURE=0.7
TOP_K=30
TOP_P=0.9
C_PUCT=0.125
ALPHA=0.5
BETA=0.9
LENGTH_SCALE=500
NUM_ROLLOUTS=4
MAX_SEARCH_COUNT=5
ROLLOUT_BUDGET=20
SAVE_DATA_TREE=True
OUTPUT_DIR="output_results_data"

# Split files directory
SPLIT_DIR="output_folds_8"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
mkdir -p  log

# Start the OmegaPRM process on each GPU with separate split files
for i in {7..10}
do
    SPLIT_FILE="$SPLIT_DIR/fold_${i}.json"
    GPU_ID=$((i-7))
    OUTPUT_FILE="$OUTPUT_DIR/results_fold_${i}.json"
    LOG_FILE_PREFIX="log/omega_prm_gpu_$GPU_ID"

    # Run the OmegaPRM process in the background on the specified GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 run_omegaprm.py \
        --question_file $SPLIT_FILE \
        --output_dir $OUTPUT_FILE \
        --model_name $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --device $DEVICE \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --top_k $TOP_K \
        --top_p $TOP_P \
        --c_puct $C_PUCT \
        --alpha $ALPHA \
        --beta $BETA \
        --length_scale $LENGTH_SCALE \
        --num_rollouts $NUM_ROLLOUTS \
        --max_search_count $MAX_SEARCH_COUNT \
        --rollout_budget $ROLLOUT_BUDGET \
        --save_data_tree $SAVE_DATA_TREE \
        --log_file_prefix $LOG_FILE_PREFIX &
done

# Wait for all processes to finish
wait

echo "All OmegaPRM processes complete."
