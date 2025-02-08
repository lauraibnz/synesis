#!/bin/bash

# List of models to test
MODELS=(
    "HuBERT"
    "Wav2Vec2"
    "CLAP"
    "AudioMAE"
    "MDuo"
    "Whisper"
    "UniSpeech"
    "XVector"
)

# Define transformations and intensities
INTENSITIES=(25 50 75 100)

# Loop through each model
for model in "${MODELS[@]}"; do
    # Loop through AddWhiteNoise intensities
    for intensity in "${INTENSITIES[@]}"; do
        tf="AddWhiteNoise${intensity}"
        
        echo "Running test for model: $model"
        echo "Transformation: $tf, Label: wps"
        echo "----------------------------------------"
        
        python -m synesis.disentanglement.downstream \
            -tf "$tf" \
            -l wps \
            -f "$model" \
            -d LibriSpeech \
            -t regression

        if [ $? -eq 0 ]; then
            echo "Successfully completed test for $model"
        else
            echo "Error occurred while testing $model"
        fi
        
        echo "----------------------------------------"
        echo ""
    done

    # Loop through PitchShift intensities
    for intensity in "${INTENSITIES[@]}"; do
        tf="PitchShift${intensity}"
        
        echo "Running test for model: $model"
        echo "Transformation: $tf, Label: wps"
        echo "----------------------------------------"
        
        python -m synesis.disentanglement.downstream \
            -tf "$tf" \
            -l wps \
            -f "$model" \
            -d LibriSpeech \
            -t regression_linear

        if [ $? -eq 0 ]; then
            echo "Successfully completed test for $model"
        else
            echo "Error occurred while testing $model"
        fi
        
        echo "----------------------------------------"
        echo ""
    done
done

echo "All tests completed"