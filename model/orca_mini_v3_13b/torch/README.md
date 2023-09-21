# ctransformers orca_mini_v3_13b

### v01 simple example 

Do CPU inference on ggml on the ggug file format through the ctransformer pythyin librabry. YOu get all the results when the program compeltes

```
python3 orca_v01.py ..//orca_mini_v3_13b.ggmlv3.q2_K.bin
```

### v02 streaming example

Do CPU inference on ggml on the ggug file format through the ctransformer pythyin librabry. YOur output is token by token.

```
python3 orca_v02.py ../../model/orca_mini_v3_13b/orca_mini_v3_13b.ggmlv3.q2_K.bin
```

### v03 streaming gpu example

Do GPU inference on ggml on the ggug file format through the ctransformer pythyin librabry. Your output is token by token.

```
python3 orca_v03.py ../../model/orca_mini_v3_13b/orca_mini_v3_13b.ggmlv3.q2_K.bin 50
```

