#!/bin/bash

# Example: Multi-Font Batch Generation
# This script demonstrates generating Sino-Nom characters with multiple fonts

echo "=========================================="
echo "Multi-Font FontDiffuser Generation"
echo "=========================================="

# Configuration
CKPT_DIR="ckpt/"
OUTPUT_DIR="data_examples/train_multi_font"
DEVICE="cuda:0"

# Use a directory containing multiple font files
# The script will automatically detect all .ttf and .otf files
TTF_DIR="ttf/"

# Alternative: Specify a single font file
# TTF_PATH="ttf/KaiXinSongA.ttf"

# Sino-Nom characters to generate
CHARACTERS="漢,字,書,法,藝,術,文,化,學,習,國,語,中,華,民,族"

# Style images
STYLE_IMAGES="data_examples/sampling/style1.jpg,data_examples/sampling/style2.jpg"

# Generation parameters
NUM_STEPS=15
GUIDANCE_SCALE=7.5
BATCH_SIZE=4
SEED=42

echo ""
echo "Configuration:"
echo "  Checkpoint: $CKPT_DIR"
echo "  Font directory: $TTF_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Characters: $CHARACTERS"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Run the batch generation
python batch_sample_evaluate.py \
    --characters "$CHARACTERS" \
    --style_images "$STYLE_IMAGES" \
    --ttf_path "$TTF_DIR" \
    --ckpt_dir "$CKPT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --num_inference_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --fp16 \
    --fast_sampling \
    --channels_last \
    --evaluate

echo ""
echo "=========================================="
echo "✓ Multi-font generation complete!"
echo "=========================================="
echo ""
echo "Check the output:"
echo "  - $OUTPUT_DIR/ContentImage/[FontName]/"
echo "  - $OUTPUT_DIR/TargetImage/[FontName]/[StyleName]/"
echo "  - $OUTPUT_DIR/summary_by_font.txt"
echo ""

# Display summary if available
if [ -f "$OUTPUT_DIR/summary_by_font.txt" ]; then
    echo "Summary by Font:"
    cat "$OUTPUT_DIR/summary_by_font.txt"
fi