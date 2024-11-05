Acces the machine with a custom [ssh_config.rw](https://github.com/alta3/infrastructure/blob/main/charlie/ssh_config.rw).

- OS: 24.04



1. Set the ssh config to access the *ssh_config.rw* file in infrastructure as follows:

    `ln -sf $HOME/git/infrastructure/charlie/ssh_config.rw $HOME/.ssh/config`

0. ssh to the enchilada host.

    `ssh enchilada.charlie`

0. Install the [a100-driver](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/?_gl=1*2hy9lp*_gcl_au*MTkxMzI5MTQyNy4xNzMwODI3NTM2&_ga=2.191448611.80666114.1730827466-91634423.1724784253)

    `sudo ubuntu-drivers install --gpgpu`
   
0. Run apt update

    `sudo apt update`

0. install the specific driver

    `sudo apt install -y nvidia-driver-535`

0. Reboot

    `sudo reboot`

0. check your drivers

    `nvidia-smi`

    ```
    Tue Nov  5 18:25:40 2024
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  NVIDIA A100 80GB PCIe          Off | 00000000:06:00.0 Off |                    0 |
    | N/A   27C    P0              43W / 300W |      0MiB / 81920MiB |      0%      Default |
    |                                         |                      |             Disabled |
    +-----------------------------------------+----------------------+----------------------+
    
    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |  No running processes found                                                           |
    +---------------------------------------------------------------------------------------+
    ```

0. Install the Cuda Compiler software.

    `sudo apt install nvidia-cuda-toolkit`

0. Run the Nvidia Cuda Compiler command.

    `nvcc -V`

    ```
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2023 NVIDIA Corporation
    Built on Fri_Jan__6_16:45:21_PST_2023
    Cuda compilation tools, release 12.0, V12.0.140
    Build cuda_12.0.r12.0/compiler.32267302_0
    ```
