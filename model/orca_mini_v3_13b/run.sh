#!/bin/bash
#
cd ~/llm/git/llama.cpp
python3 -m pip install numpy

./convert-llama-ggmlv3-to-gguf.py \
    --input  ../../model/orca_mini_v3_13b/orca_mini_v3_13b.ggmlv3.q8_0.bin \
    --output ../../model/orca_mini_v3_13b/orca_mini_v3_13b.gguf.q8_0.bin 

# defaulting to low cpu numbers that will work everywhere
./main -ngl 0 --threads 14 \
    --model  ../../model/orca_mini_v3_13b/orca_mini_v3_13b.gguf.q8_0.bin \
    --file ../../prompt/instruct/ansible.txt
