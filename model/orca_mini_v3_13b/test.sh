#!/bin/bash

# change into this script's dir
cd -P -- "$(dirname -- "$0")"

# Install `numpy`, required for conversion script:
python3 -m pip install numpy

# Convert from ggmlv3 to gguf format:
../../git/llama.cpp/convert-llama-ggml-to-gguf.py \
    --input  model/orca_mini_v3_13b.ggmlv3.q8_0.bin \
    --output model/orca_mini_v3_13b.gguf.q8_0.bin

# Run inference with the converted model `llama.cpp`
# defaulting to low cpu numbers that will work everywhere
../../git/llama.cpp/main -ngl 0 --threads 14 \
    --model model/orca_mini_v3_13b.gguf.q8_0.bin \
    --file ../../prompt/instruct/ansible.txt
