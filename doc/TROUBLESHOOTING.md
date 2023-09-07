
## Apt / Nvidia

Observed error:
```
CUDA error 222 at /home/runner/work/ctransformers/ctransformers/models/ggml/ggml-cuda.cu:6045: the provided PTX was compiled with an unsupported toolchain.
```

Fix:

```bash
bash ansible/nvidia.sh
```
