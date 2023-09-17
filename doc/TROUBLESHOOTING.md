
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
