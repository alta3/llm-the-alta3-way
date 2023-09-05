
## Apt / Nvidia




```bash
sudo apt --purge remove -y "*cublas*" "cuda*" "*nvidia*"
sudo apt clean
sudo apt autoremove -y

export NDV=535 # NVIDIA DRIVER VERSION
sudo apt install -y \
    cuda-drivers-$NDV \
    libnvidia-compute-$NDV-server \
    libnvidia-compute-$NDV \
    nvidia-cuda-dev \
    nvidia-cuda-toolkit
```


