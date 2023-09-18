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

0.  ## **torch.stack()** (tensors, dim=0, *, out=None)

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

     **Optional Keywords:**

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

0. ## **torch.cat** (tensors, dim=0, *, out=None)

     **Parameters:**
   
     - tensors (sequence of Tensors) – any python sequence of tensors of the same type. Non-empty tensors provided must have the same shape, except in the cat dimension.
     - dim (int, optional) – the dimension over which the tensors are concatenated

    **Optional Keyword:**
   
     - out (Tensor, optional) – the output tensor.

0. ## **torch.multinomial()**  (input, num_samples, replacement=False, *, generator=None, out=None)

     > Returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.

     **Parameters:**
     
     - input (Tensor) – the input tensor containing probabilities
     - num_samples (int) – number of samples to draw
     - replacement (bool, optional) – whether to draw with replacement or not

     **Optional Keywords:**

     - generator (torch.Generator, optional) – a pseudorandom number generator for sampling
     - out (Tensor, optional) – the output tensor.

0. ## **nn.Module:** (*args, **kwargs)

    > Base class for all neural network modules.
    
    ```
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))
    ```

0. ## **nn.Linear()**

   > Applies alinear transformation to the incoming data: y = xA^T + b
   
   **Parameters:**
   
    - in_features (int) – size of each input sample
    - out_features (int) – size of each output sample
    - bias (bool) – If set to False, the layer will not learn an additive bias. Default: True

   **Variables** 

    - weight (torch.Tensor): the learnable weights of the module of shape (out_features,in_features). 
    - bias: the learnable bias of the module of shape (out_features). 

0. ## **@ or torch.matmul()** (input, other, *, out=None)

   > This is an OPERATOR that allows from matrix multiplication

   **Parameters:**

   - input (Tensor) – the first tensor to be multiplied
   - other (Tensor) – the second tensor to be multiplied
   
0. ## **masked_fill_()**  (mask, value)

    > Fills elements of self tensor with value where mask is True. The shape of mask must be broadcastable with the shape of the underlying tensor.

    **Parameters:**

   - mask (BoolTensor) – the boolean mask
   - value (float) – the value to fill in with



   
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
