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

# Loop through each model
for model in "${MODELS[@]}"; do
    echo "Running test for model: $model"
    echo "----------------------------------------"
    
    python -m synesis.invariance.covariate_shift \
        -tf AddReverb \
        -f "$model" \
        -d LibriSpeech \
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