#!/bin/bash

# Example script for running batch FontDiffuser generation and evaluation
# Modify paths and parameters according to your setup

echo "=========================================="
echo "FontDiffuser Batch Generation & Evaluation"
echo "=========================================="

# Configuration
CKPT_DIR="ckpt/"
OUTPUT_DIR="data_examples/train"
TTF_PATH="ttf/KaiXinSongA.ttf"
DEVICE="cuda:0"

# Sino-Nom characters to generate
# You can modify this list or use a file
CHARACTERS="漢,字,書,法,藝,術,文,化,學,習"

# Style images
# Can be comma-separated paths or a directory
STYLE_IMAGES="data_examples/sampling/style1.jpg,data_examples/sampling/style2.jpg"

# Generation parameters
NUM_STEPS=20
GUIDANCE_SCALE=7.5
BATCH_SIZE=4
SEED=42

# Optimization flags
USE_FP16="--fp16"
FAST_SAMPLING="--fast_sampling"
CHANNELS_LAST="--channels_last"

# Optional: Enable xformers if installed
# XFORMERS="--enable_xformers"
XFORMERS=""

# Optional: Enable torch.compile (PyTorch 2.0+)
# COMPILE="--compile"
COMPILE=""

# Evaluation options
EVALUATE="--evaluate"
# GROUND_TRUTH_DIR="path/to/ground_truth"  # Uncomment if you have ground truth
GROUND_TRUTH_DIR=""
COMPUTE_FID=""  # Add "--compute_fid" if you want FID score

# W&B options (uncomment to enable)
# USE_WANDB="--use_wandb"
# WANDB_PROJECT="--wandb_project fontdiffuser-sino-nom"
# WANDB_RUN="--wandb_run_name experiment_$(date +%Y%m%d_%H%M%S)"
USE_WANDB=""
WANDB_PROJECT=""
WANDB_RUN=""

echo ""
echo "Configuration:"
echo "  Checkpoint: $CKPT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Characters: $CHARACTERS"
echo "  Styles: $STYLE_IMAGES"
echo "  Batch size: $BATCH_SIZE"
echo "  Inference steps: $NUM_STEPS"
echo ""

# Run the batch generation and evaluation
python batch_sample_evaluate.py \
    --characters "$CHARACTERS" \
    --style_images "$STYLE_IMAGES" \
    --ckpt_dir "$CKPT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --ttf_path "$TTF_PATH" \
    --device "$DEVICE" \
    --num_inference_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    $USE_FP16 \
    $FAST_SAMPLING \
    $CHANNELS_LAST \
    $XFORMERS \
    $COMPILE \
    $EVALUATE \
    ${GROUND_TRUTH_DIR:+--ground_truth_dir "$GROUND_TRUTH_DIR"} \
    $COMPUTE_FID \
    $USE_WANDB \
    $WANDB_PROJECT \
    $WANDB_RUN

echo ""
echo "=========================================="
echo "✓ Batch generation complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Check results.json for detailed metrics"
echo ""

# Optional: Display directory structure
if command -v tree &> /dev/null; then
    echo "Output directory structure:"
    tree -L 3 "$OUTPUT_DIR"
else
    echo "Install 'tree' command to see directory structure"
fi