# `falcon-40b-instruct`

## Primary sources

- Original model: [TheBloke/falcon-40b-instruct](https://huggingface.co/TheBloke/falcon-40b-instruct)
  <!-- This is a quantized model but it specifically mentions that it does NOT work with llama.cpp which means we can't use it, correct? -->
<!-- - Quantized: [tiiuae/falcon-40b-instruct-GGML](https://huggingface.co/tiiuae/falcon-40b-instruct-GGML) -->

## Local copies of Model Cards

- Original model: [README_ttiuae.md](./README_tiiuae.md)
  <!-- Same problem as above -->
<!-- - Quantized: [README_TheBloke.md](./README_TheBloke.md) -->

## Quickstart

```bash
ansible-playbook model/tiiuae/falcon-40b-instruct/install.yml
```

```bash
./model/tiiuae/falcon-40b-instruct/test.sh
```

