# `orca_mini_v3_13b`

## Primary sources

- [psmathur/orca_mini_v3_13b](https://huggingface.co/psmathur/orca_mini_v3_13b)
- [TheBloke/orca_mini_v3_13B-GGML](https://huggingface.co/TheBloke/orca_mini_v3_13B-GGML)

## Local copies of Model Cards

- [psmathur_README.md](./psmathur_README.md)
- [TheBloke_README.md](./TheBloke_README.md)

## Inference example

```bash
cd llm/git/llama.cpp
python3 -m pip install numpy

./convert-llama-ggmlv3-to-gguf.py \
    --input  ../../model/orca_mini_v3_13b/orca_mini_v3_13b.ggmlv3.q8_0.bin \
    --output ../../model/orca_mini_v3_13b/orca_mini_v3_13b.ggug.q8_0.bin 

./main -ngl <LAYERS> --threads 14 \
    --model  ../../model/orca_mini_v3_13b/orca_mini_v3_13b.ggug.q8_0.bin \
    --file ../../prompt/instruct/ansible.txt"
```

        



