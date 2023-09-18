# An Attempt to make a mini-docs for our used PyTorch functions

 > Note about the Import structure: We breakout Torch sublibraries that are specifically dealing with nn-Nueral Networks and F-Functions

       ```
       import torch
       import torch.nn as nn
       from torch.nn import functional as F
       ```

# Commonly Used Torch Functions (Arranged by usage in Infinite Shakespeare)

1. ## **torch.tensor()** (data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) 

    **Parameters:**
  
    - **data:** Initial data for the tensor. Can be a list, tuple, NumPy ndarray, scalar, and other types.
       
    **Optional Keywords:**
  
    - **dtype:** datatype of returned tensor, default: None will infer type from `data`
    - **device:** select CPU or GPU for tensor location, default will take device from `data`
    - **requires_grad:** should autograd record operations on the returned tensor. Default: False
    - **pin_memory:** If set, returned tensor would be allocated in the pinned memory. Works only for CPU tensors. Default: False.


0. ## **torch.randint()** (low=0, high, size, \*, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)

  **Parameters:**

  - low (int) – Lowest integer to be drawn from the distribution. Default: 0.
  - high (int) – One above the highest integer to be drawn from the distribution.
  - size (tuple) – a tuple defining the shape of the output tensor.

  **Optional Keywords:**

  - generator (torch.Generator) – a pseudorandom number generator for sampling
  - out (Tensor) – the output tensor.
  - dtype (torch.dtype) – if None, this function returns a tensor with dtype torch.int64.
  - layout (torch.layout) – the desired layout of returned Tensor. Default: torch.strided.
  - device (torch.device) – the desired device of returned tensor. Default: if None, uses the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
  - requires_grad (bool) – If autograd should record operations on the returned tensor. Default: False.

  ## **torch.stack()** (tensors, dim=0, *, out=None)

  > Note: All Tensors need to be the same size

  **Parameters:**

  - tensors (sequence of Tensors) – sequence of tensors to concatenate
  - dim (int) – dimension to insert. Has to be between 0 and the number of dimensions of concatenated tensors (inclusive)

  **Optional Keywords:**

  - out (Tensor) – the output tensor.

0. ## **torch.tril()** (input, diagonal=0, *, out=None)

    **Parameters:**

    - input (Tensor) – the input tensor.
    - diagonal (int, optional) – the diagonal to consider

    **Keyword Arguments:**

    - out (Tensor, optional) – the output tensor.

    Example:

    ```
    >>> a = torch.randn(3, 3)
    >>> a
    tensor([[-1.0813, -0.8619,  0.7105],
            [ 0.0935,  0.1380,  2.2112],
            [-0.3409, -0.9828,  0.0289]])
    >>> torch.tril(a)
    tensor([[-1.0813,  0.0000,  0.0000],
            [ 0.0935,  0.1380,  0.0000],
            [-0.3409, -0.9828,  0.0289]])
    ```

0. ## **torch.cat**









torch.arrange
torch.multinomial
torch.long
nn.Module:
nn.Linear:
@ or torch.matmul:
masked_fill:
F.Softmax:
nn.Dropout
nn.ModuleList:
nn.Sequential
nn.Embedding
nn.ReLU
nn.LayerNorm
F.cross_entropy
torch.optim.AdamW
.to(device) / torch.cuda
