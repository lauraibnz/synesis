#!/bin/bash

# List of models to test
MODELS=(
    ResNet18_ImageNet
    ResNet34_ImageNet
    ResNet50_ImageNet
    ResNet101_ImageNet
    ViT_b_16_ImageNet
    ViT_l_16_ImageNet
    ViT_b_32_ImageNet
    ViT_l_32_ImageNet
    DINOv2_small
    DINOv2_base
    DINOv2_large
    DINO
    SimCLR
    ViT_MAE
    CLIP
    IJEPA
)

# Loop through each model
for model in "${MODELS[@]}"; do
    echo "Running test for model: $model"
    echo "----------------------------------------"
    
    python -m synesis.invariance.covariate_shift \
        -tf JPEGCompression \
        -f "$model" \
        -d ImageNet \
        -b 1 \
        -p 1

    # Check if command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully completed test for $model"
    else
        echo "Error occurred while testing $model"
    fi
    
    echo "----------------------------------------"
    echo ""
done

echo "All tests completed"