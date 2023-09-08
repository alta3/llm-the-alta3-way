# `falcon-40b-instruct`

## Primary sources

- Original model: [tiiuae/falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct)
  <!-- This is a quantized model but it specifically mentions that it does NOT work with llama.cpp which means we can't use it, correct? -->
<!-- - Quantized: [TheBloke/falcon-40b-instruct-GGML](https://huggingface.co/TheBloke/falcon-40b-instruct-GGML) -->

## Local copies of Model Cards

- Original model: [README_ttiuae.md](./README_tiiuae.md)
- Quantized: [README_TheBloke.md](./README_TheBloke.md)

## Quickstart

```bash
ansible-playbook model/orca_mini_v3_13b/install.yml
```

```bash
./model/orca_mini_v3_13b/test.sh
```
https://huggingface.co/tiiuae/falcon-40b-instruct
