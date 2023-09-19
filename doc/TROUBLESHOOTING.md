
## Apt / Nvidia

Observed error:
```
CUDA error 222 at /home/runner/work/ctransformers/ctransformers/models/ggml/ggml-cuda.cu:6045: the provided PTX was compiled with an unsupported toolchain.
```

Fix:

```bash
bash ansible/nvidia.sh
```

## llama.cpp ggml-cuda.cu

Error:
```
Log start                                                                                       
main: build = 1253 (111163e)                                                                    
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu                  
main: seed  = 1694974558                                                                        
                                                                                                
CUDA error 3 at ggml-cuda.cu:5522: initialization error                                         
current device: 21936
```

This has been observed when running an older version of cuda driver/toolkit and tring to run `llama.cpp` on a 
SMX device. In order to fix this error, reinstall drivers with a higher version (e.g. 535) and reboot.

- `cd ~/llm/git/llama.cpp`
- `make clean`
- `LLAMA_CUBLAS=1 make -j12`
- `./main`

<details>
  <summary>Nvidia SMI - Version Details</summary>
<pre><code>+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-80GB          On  | 00000000:00:05.0 Off |                   On |
| N/A   46C    P0              62W / 500W |      0MiB / 81920MiB |     N/A      Default |
|                                         |                      |              Enabled |
+-----------------------------------------+----------------------+----------------------+
+---------------------------------------------------------------------------------------+
| MIG devices:                                                                          |
+------------------+--------------------------------+-----------+-----------------------+
| GPU  GI  CI  MIG |                   Memory-Usage |        Vol|      Shared           |
|      ID  ID  Dev |                     BAR1-Usage | SM     Unc| CE ENC DEC OFA JPG    |
|                  |                                |        ECC|                       |
|==================+================================+===========+=======================|
|  No MIG devices found                                                                 |
+---------------------------------------------------------------------------------------+                        
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+</code></pre>
</details>




## ninja in python project does not work under systemd.

Steps to failure:
 - Install dependencies of a python project in a venv
 - Setup a system unit file to run the service without 'activate' (this should work)
 - See ninja blow up:

```
Sep 19 03:47:23 a100-1 systemd[1]: Started webui.                                                                                                                                                 
Sep 19 03:47:27 a100-1 python3[239464]: Traceback (most recent call last):                                                    
<snip>
Sep 19 03:47:27 a100-1 python3[239464]:   File "/home/ubuntu/llm/git/webui/venv/lib/python3.10/site-packages/exllamav2/ext.py", line 118, in <module>
Sep 19 03:47:27 a100-1 python3[239464]:     exllamav2_ext = load \                                                                                                                                
Sep 19 03:47:27 a100-1 python3[239464]:   File "/home/ubuntu/llm/git/webui/venv/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 1284, in load
Sep 19 03:47:27 a100-1 python3[239464]:     return _jit_compile(                                                                                                                                  
Sep 19 03:47:27 a100-1 python3[239464]:   File "/home/ubuntu/llm/git/webui/venv/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 1509, in _jit_compile
Sep 19 03:47:27 a100-1 python3[239464]:     _write_ninja_file_and_build_library(                 
Sep 19 03:47:27 a100-1 python3[239464]:   File "/home/ubuntu/llm/git/webui/venv/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 1593, in _write_ninja_file_and_build_library
Sep 19 03:47:27 a100-1 python3[239464]:     verify_ninja_availability()                          
Sep 19 03:47:27 a100-1 python3[239464]:   File "/home/ubuntu/llm/git/webui/venv/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 1649, in verify_ninja_availability
Sep 19 03:47:27 a100-1 python3[239464]:     raise RuntimeError("Ninja is required to load C++ extensions")
Sep 19 03:47:27 a100-1 python3[239464]: RuntimeError: Ninja is required to load C++ extensions
```

Fix:
This is happening because ninja libray is looking for `ninja` to be in the `PATH`.
Updating `PATH` for the unit file didn't seem to work.
So... make the unit file source the venv (🤮)

```
# sourcing venv required due to ninja pkg looking for itself in PATH
WorkingDirectory=/home/ubuntu/llm/git/webui
ExecStart=/bin/sh -c '. venv/bin/activate && python3 server.py'
```
