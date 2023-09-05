
## Apt / Nvidia

Observed error:
```
CUDA error 222 at /home/runner/work/ctransformers/ctransformers/models/ggml/ggml-cuda.cu:6045: the provided PTX was compiled with an unsupported toolchain.
```

```bash
{
  sudo apt --purge remove -y "*cublas*" "cuda*" "*nvidia*"
  sudo apt clean
  sudo apt autoremove -y
  
  DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
  sudo -E apt-get install -y nvidia-cuda-toolkit
}
```


