
## Apt / Nvidia

```bash
sudo apt --purge remove -y "*cublas*" "cuda*" "*nvidia*"
sudo apt clean
sudo apt autoremove -y

DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
sudo apt-get install -y nvidia-cuda-toolkit
```


