
## Apt / Nvidia

Observed error:
```
CUDA error 222 at /home/runner/work/ctransformers/ctransformers/models/ggml/ggml-cuda.cu:6045: the provided PTX was compiled with an unsupported toolchain.
```

Fix:

```bash
bash ansible/nvidia.sh
```

## ggml-cuda.cu

```
Log start                                                                                       
main: build = 1253 (111163e)                                                                    
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu                  
main: seed  = 1694974558                                                                        
                                                                                                
CUDA error 3 at ggml-cuda.cu:5522: initialization error                                         
current device: 21936
```
