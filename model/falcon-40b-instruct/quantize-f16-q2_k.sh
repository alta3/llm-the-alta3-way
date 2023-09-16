# This script expects a f16 base model to quantize
# Actually perform the quantization
# change into this script's dir
cd -P -- "$(dirname -- "$0")"

../../git/llama.cpp/quantize \
    model/ggml-model-f16.gguf \
    model/falcon-40b-instruct.gguf.q2_k.bin \
    q2_k
    
# Test quantization efficacy by running a prompt. Success will result in response tokens being generated
../../git/llama.cpp/main -ngl 83 --threads 22 \
    --model model/falcon-40b-instruct.gguf.q2_k.bin \
    --file ../../prompt/instruct/ansible.txt
