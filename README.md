# `alta3/llm-the-alta3-way`

```
├── playbook     # playbooks
├── prompt       # prompts
├── torch        # pyTorch
└── training     # training data
    └── instruct # instruct style training data
```

## Assumptions

- Ubuntu 20.04 taret host (e.g. Lamba Labs)
- Tested with Nvidia H100

## Roadmap

### Frameworks

- [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- [cmp-nct/ggllm.cpp](https://github.com/cmp-nct/ggllm.cpp)
- [ ] [turboderp/exllama](https://github.com/turboderp/exllama)

### Tools

- [simonw/llm](https://github.com/simonw/llm) 
   ```bash
   llm --help; llm --version
   ```
- [simonw/ttok](https://github.com/simonw/ttok) 
   ```bash
   ttok --help; ttok --version
   ```
- [aristocratos/bpytop](https://github.com/aristocratos/bpytop)
   ```bash
   bpytop
   ```
- [wookayin/gpustat](https://github.com/wookayin/gpustat)
   ```bash
   gpustat --interval 1 --show-all
   ```
- [bodaay/HuggingFaceModelDownloader](https://github.com/bodaay/HuggingFaceModelDownloader)
   ```bash
   hfdownloader --help
   ```
- [golang 1.20.5](https://go.dev/)
   ```bash
   go help; go version; which go
   ```

### Models

- [Falcon 40B Instruct](https://huggingface.co/TheBloke/falcon-40b-instruct-GGML) with ggllm.cpp: 
   ```bash
   ansible-playbook playbook/falcon-40b-instruct.yml
   ```
- [Orca Mini v2 13B](https://huggingface.co/TheBloke/orca_mini_v2_13b-GGML) with llama.cpp
   ```bash
   ansible-playbook playbook/orca_mini_v2_13b.yml
   ```
