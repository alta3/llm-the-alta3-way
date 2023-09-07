#!/bin/bash

# change into this script's dir
cd -P -- "$(dirname -- "$0")"

# Run inference with the converted model `llama.cpp`
# test with low cpu numbers, no gpu, 10 tokens in order to get numbers
../../git/llama.cpp/main -ngl 0 --threads 8 --n-predict 10 \
    --model model/llama-2-70b-orca-200k.Q2_K.gguf \
    --file ../../prompt/instruct/ansible.txt
