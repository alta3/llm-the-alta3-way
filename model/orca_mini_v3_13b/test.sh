#!/bin/bash

# change into this script's dir
cd -P -- "$(dirname -- "$0")"
python3 -m pip install numpy

../../git/llama.cpp/convert-llama-ggmlv3-to-gguf.py \
    --input  orca_mini_v3_13b.ggmlv3.q8_0.bin \
    --output orca_mini_v3_13b.gguf.q8_0.bin

# defaulting to low cpu numbers that will work everywhere
../../git/llama.cpp/main -ngl 0 --threads 14 \
    --model orca_mini_v3_13b.gguf.q8_0.bin \
    --file ../../prompt/instruct/ansible.txt