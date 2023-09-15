#!/bin/bash

# change into this script's dir
cd -P -- "$(dirname -- "$0")"

# Install `numpy`, `transformers` `torch` required for conversion script
python3 -m pip install \
    --pre 'torch>=2.1.0dev' torchvision torchaudio numpy  \
    --index-url https://download.pytorch.org/whl/nightly/cu118
python3 -m pip install transformers


# Convert from ggmlv3 to gguf format:
../../git/llama.cpp/convert-falcon-hf-to-gguf.py \
    model 1

# Run inference with the converted model `llama.cpp`
# test with low cpu numbers, no gpu, 10 tokens
../../git/llama.cpp/main -ngl 0 --threads 8 --n-predict 10 \
    --model model/ggml-model-f16.gguf \
    --file ../../prompt/instruct/ansible.txt

