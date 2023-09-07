#!/bin/bash

# change into this script's dir
cd -P -- "$(dirname -- "$0")"

# Install `numpy`, required for conversion script:
python3 -m pip install numpy

# Convert from ggmlv3 to gguf format:
../../git/llama.cpp/convert-falcon-hf-to-gguf.py \
    model/falcon-40b-instruct 1

# Run inference with the converted model `llama.cpp`
# test with low cpu numbers, no gpu, 10 tokens
../../git/llama.cpp/main -ngl 0 --threads 8 --n-predict 10 \
    --model model/falcon-40b-instruct.gguf \
    --file ../../prompt/instruct/ansible.txt
