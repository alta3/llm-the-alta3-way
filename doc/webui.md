## Setup 

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm5.4.2
python3 -m pip install llama-cpp-python ctransformers
```

## Link models

```
cd ~/llm/git/webui/models
ln -s ~/llm/model/Llama-2-70B-Orca-200k/model/* .
```

## Run

```
python3 server.py
```

## Reverse Proxy

```bash
sudo caddy reverse-proxy --from https://{{ fqdn }} --to 127.0.0.1:7860                       
```
