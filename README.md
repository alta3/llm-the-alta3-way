# `alta3/llm-the-alta3-way`

```
├── ansible      # common playbooks and roles
├── doc          # troubleshooting and other misc documentation
│   └── provider # cloud gpu provider details and fix-up scripts
├── model        # models with supporting fetch/clean/unsplit scripts
└── prompt       # prompts for testing/demonstration
```

## Assumptions

- Ubuntu 20.04 or 22.04 target host
- Tested with Nvidia H100, A100, A40

## Quickstart

1. Every cloud service PROVIDER requires specific configuration in order to complete this quickstart. As soon as you have connected to the provider's host, [find your CLOUD PROVIDER on this list](doc/provider/README.md), read the README found there and run the tasks specified in that README. If you skip this step, things will NOT work.

0. Exit the provider's machine.

0. ssh back into the provider as ubuntu.

0. Setup ansible and clone this repo

   ```bash
   {
     git clone https://github.com/alta3/llm-the-alta3-way.git
     cd llm-the-alta3-way
     bash ansible/nvidia.sh
     python3 -m pip install --upgrade --user pip
     python3 -m pip install --user ansible
   }
   ```

0. System is required because cuda-drivers are reinstalled.

   ```bash
   sudo systemctl reboot
   ```

0. Run `nvcc --version` and `nvidia-smi` to verify versions.

   ```bash
   nvcc --version
   nvidia-smi
   ```

0. Select a model and Run (see models section for specific playbooks)

   ```bash
   cd ~/llm-the-alta3-way/
   ansible-playbook model/{{ model }}/install.yml
   bash ~/llm/model/{{ model }}/test.sh
   ```

0. Want to try again?  This directory structure is created to make that action really easy. Perform the following to reset your machine:

  rm -r ~/llm
  rm -r ~/llm-the-alta3-way

## Models

- [x] [Llama2 70B Orca 200k GGUF](https://huggingface.co/TheBloke/Llama-2-70B-Orca-200k-GGUF)
   ```bash
   ansible-playbook model/Llama-2-70B-Orca-200k/install.yml
   ```

- [x] [Orca Mini v3 13B GGML](https://huggingface.co/TheBloke/orca_mini_v3_13b-GGML)
   ```bash
   ansible-playbook model/orca_mini_v3_13b/install.yml
   ```

- [x] [Falcon 40B Instruct](https://huggingface.co/tiiuae/falcon-40b-instruct): 
   ```bash
   ansible-playbook model/falcon-40b-instruct/install.yml
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
│   └── prompt         # prompts for testing/demonstration      
└── llm-the-alta3-way  # this repo checked out
```


### Model Loaders Frameworks

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
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)
   ```bash
   cd ~/llm/git/webui
   source venv/bin/activate
   python3 server.py
   ```

### Additional dependencies

- [jllllll/llama-cpp-python-cuBLAS-wheels](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels)

### Deprecated

- [cmp-nct/ggllm.cpp](https://github.com/cmp-nct/ggllm.cpp) - [Falcon support added to llama.cpp](https://github.com/ggerganov/llama.cpp/issues/1602)
