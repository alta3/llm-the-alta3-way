../../git/llama.cpp/quantize \
    model/ggml-model-f16.gguf \
    model/falcon-40b-instruct.gguf.q4_k_m.bin \
    q4_k_m

../../git/llama.cpp/main -ngl 100 --threads 21 \
    --model model/falcon-40b-instruct.gguf.q4_k_m.bin \
    --file ../../prompt/instruct/ansible.txt
