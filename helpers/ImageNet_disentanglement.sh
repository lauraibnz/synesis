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

# Define labels and base transformations
LABELS=(hue saturation brightness)
BASE_TRANSFORMATIONS=(Hue Saturation Brightness)
INTENSITIES=(25 50 75 100)

# Loop through each model
for model in "${MODELS[@]}"; do
    # Loop through each base transformation
    for base_tf in "${BASE_TRANSFORMATIONS[@]}"; do
        # Loop through intensities
        for intensity in "${INTENSITIES[@]}"; do
            # Construct full transformation name
            tf="${base_tf}Shift${intensity}"
            
            # Loop through each label
            for label in "${LABELS[@]}"; do
                # Skip if base transformation matches label (case-insensitive)
                if [ "${base_tf,,}" = "${label,,}" ]; then
                    continue
                fi
                
                echo "Running test for model: $model"
                echo "Transformation: $tf, Label: $label"
                echo "----------------------------------------"
                
                python -m synesis.disentanglement.downstream \
                    -tf "$tf" \
                    -l "$label" \
                    -f "$model" \
                    -d ImageNet \
                    -t regression_linear

                # Check if command was successful
                if [ $? -eq 0 ]; then
                    echo "Successfully completed test for $model"
                else
                    echo "Error occurred while testing $model"
                fi
                
                echo "----------------------------------------"
                echo ""
            done
        done
    done
done

echo "All tests completed"