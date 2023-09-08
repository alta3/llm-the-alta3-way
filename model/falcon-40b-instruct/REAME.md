# `falcon-40b-instruct`

## Primary sources

- Original model: [TheBloke/Llama-2-70B-Orca-200k](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GGUF)
  <!-- This is a quantized model but it specifically mentions that it does NOT work with llama.cpp which means we can't use it, correct? -->
<!-- - Quantized: [TheBloke/falcon-40b-instruct-GGML](https://huggingface.co/TheBloke/falcon-40b-instruct-GGML) -->

## Local copies of Model Cards

- Original model: [README_ttiuae.md](./README_tiiuae.md)
  <!-- Same problem as above -->
<!-- - Quantized: [README_TheBloke.md](./README_TheBloke.md) -->

## Quickstart

```bash
ansible-playbook model/TheBloke/Llama-2-70B-Orca-200k/install.yml
```

```bash
./model/TheBloke/Llama-2-70B-Orca-200k/test.sh
```

