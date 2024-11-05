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

    > This will take approx. 30 Minutes to complete.
    >
    > **IMPORTANT** -- As of 4/11 -- An issue with the Paperspace VM was noted to case an error which impacts the **APT** Package Manager. Exact steps to correct were not recorded, however your instructor will know what to do. If/When you encounter this issue, please inform your instructor.

0. System reboot is required because cuda-drivers are reinstalled.

   ```bash
   sudo systemctl reboot
   ```

    > It may take a few minutes to restart the VM. If your SSH connection fails, try again after a few minutes.

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

    > It will take approx. 1.5 hours for this playbook to complete. The reason for this is needing to download the model to the Paperspace VM which takes quite a while. It may appear the Ansible Playbook has stalled, but I assure you -- It has not.

0. Use TMUX and split your screen into 3 panes

    To perform a Vertical Split -- **CTRL+B** - **HOK** - **%**
    To perform a Horizontal Split -- **CTRL+B** - **HOK** - **"**

    > - TMUX PANE #1 - A Panel reserved for running LLAMA-2 in Server Mode.
    > - TMUX PANE #2 - A Panel reserved for running Caddy in Reverse Proxy Mode
    > - TMUX PANE #3 - Command Line operations

0. Want to try again?  This directory structure is created to make that action really easy. rm the following directories to reset your machine:

    ```
    rm -r ~/llm
    rm -r ~/llm-the-alta3-way
    ```
