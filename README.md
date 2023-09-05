# `alta3/llm-the-alta3-way`

```
├── docs       # troubleshooting and other misc documentation
├── model      # models with supporting fetch/clean/unsplit scripts
├── playbook   # playbooks for deploying models and dependencies
├── prompt     # prompts for testing/demonstration
├── providers  # cloud gpu provider details and fix-up scripts
├── torch      # steps and resources for interacting with models from python
└── training   # training data
```

## Assumptions

- Ubuntu 20.04 or 22.04 target host
- Tested with Nvidia H100, A100

## Quickstart

Baseline pre-playbook installs and updates

```bash
{
  sudo apt update 
  DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
  sudo -E apt-get install -y python3-pip python3-venv git
  python3 -m pip install --upgrade --user pip
  python3 -m pip install --user ansible
  git clone https://github.com/alta3/llm-the-alta3-way
  cd llm-the-alta3-way
}
```

Optional:

```bash
sudo apt install -y cuda-drivers
sudo reboot
```

Run (see models section for specific playbooks)

```bash
# may need to source .profile or .bashrc to add ~/.local/bin/ to $PATH
ansible-playbook playbook/<model>.yml
```

### Models

- [x] [Llama2 70B Orca 200k GGUF](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GGUF)
   ```bash
   ansible-playbook playbook/Llama-2-70B-Orca-200k.yml
   ```

- [x] [Orca Mini v3 13B GGML](https://huggingface.co/TheBloke/orca_mini_v3_13b-GGML)
   ```bash
   ansible-playbook playbook/orca_mini_v3_13b.yml
   ```

- [x] [Falcon 40B Instruct](https://huggingface.co/tiiuae/falcon-40b-instruct): 
   ```bash
   ansible-playbook playbook/falcon-40b-instruct.yml
   ```

- [ ] [CodeLlama](https://github.com/facebookresearch/codellama)

### `llm` directory

Deployed by this repo's base role, this directory structure is a non-git directory named `llm` for development and work with deployed models.

```
├── llm                # llm working directory
│   ├── bin            # installed binaries (e.g. hfdownloader)
│   ├── dataset        # <future use>
│   ├── git            # installed git repos
│   │   └── llama.cpp  # inference of LLaMA model in pure C/C++
│   ├── model          # deployed models
│   ├── prompt         # prompts for testing/demonstration      
│   └── torch          # steps and resources for interacting with models from python
└── llm-the-alta3-way  # this repo checked out
```


### Frameworks

- [x] [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- [x] [marella/ctransformers](https://github.com/marella/ctransformers)
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


### Deprecated

- [cmp-nct/ggllm.cpp](https://github.com/cmp-nct/ggllm.cpp) - [Falcon support added to llama.cpp](https://github.com/ggerganov/llama.cpp/issues/1602)
