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

0. ## **F.Softmax()**

    <img src="https://www.gstatic.com/education/formulas2/553212783/en/softmax_function.svg"/>

    **Returns:**
    - a Tensor of the same dimension and shape as the input with values in the range [0, 1]

    **Parameters:**
    - dim (int) – A dimension along which Softmax will be computed (so every slice along dim will sum to 1).

   
0. ## **nn.Dropout**

    > During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a
      Bernoulli distribution. Each channel will be zeroed out independently on every forward call.
    
    ***Parameters:***

    - p (float) – probability of an element to be zeroed. Default: 0.5
    - inplace (bool) – If set to True, will do this operation in-place. Default: False



0. ## **nn.ModuleList:**

   > Holds submodules in a list.

   Example:

   ```
   class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x
   ```

0. ## **nn.Sequential()** (Module)

    > A sequential container. Modules will be added to it in the order they are passed in the constructor.

    > NOTE: What’s the difference between a Sequential and a torch.nn.ModuleList? A ModuleList is exactly
       what it sounds like–a list for storing Module s! On the other hand, the layers in a Sequential are
       connected in a cascading way.    


0. ## **nn.Embedding()** (num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)

      > This module is often used to store word embeddings and retrieve them using indices.
      The input to the module is a list of indices, and the output is the corresponding word embeddings.
      
    **Parameters:**

    - num_embeddings (int) – size of the dictionary of embeddings
    - embedding_dim (int) – the size of each embedding vector
    - padding_idx (int, optional) – If specified, the entries at padding_idx do not contribute to the gradient
    - max_norm (float, optional) – If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm.
    - norm_type (float, optional) – The p of the p-norm to compute for the max_norm option. Default 2.
    - scale_grad_by_freq (bool, optional) – If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default False.
    - sparse (bool, optional) – If True, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for more details regarding sparse gradients.
      
    **Variables:**
   
    - weight (Tensor) – the learnable weights of the module of shape (num_embeddings, embedding_dim)

0. ## **nn.ReLU** 

    > Applies the rectified linear unit function element-wise:

    ```
    ReLU(x)=(x)^+ = max(0,x)
    ```

    **Parameters:**

    - inplace (bool) – can optionally do the operation in-place. Default: False


0. ## **nn.LayerNorm**  (normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None

    > Applies Layer Normalization over a mini-batch of inputs

    <src img="https://miro.medium.com/v2/resize:fit:1040/0*qN-QGSHiY85obQfj" />

    **Parameters:**

   - normalized_shape (int or list or torch.Size) –
   - input shape from an expected input of size. If a single integer is used, it is treated as a singleton list, and this module will normalize over the last dimension which is expected to be of that specific size.
   - eps (float) – a value added to the denominator for numerical stability. Default: 1e-5
   - elementwise_affine (bool) – a boolean value that when set to True, this module has learnable per-element affine parameters initialized to ones (for weights) and zeros (for biases). Default: True.

    **Variables:**

   - weight – the learnable weights of the module of shape 'normalized_shape' when 'elementwise_affine' is set to True. The values are initialized to 1.
   - bias – the learnable bias of the module of shape 'normalized_shape' when 'elementwise_affine' is set to True. The values are initialized to 0.

0. ## **F.cross_entropy** (input, target, weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean', label_smoothing=0.0)

    **Parameters:**

   - input (Tensor) – Predicted unnormalized logits; see Shape section below for supported shapes.
   - target (Tensor) – Ground truth class indices or class probabilities; see Shape section below for supported shapes.
   - weight (Tensor, optional) – a manual rescaling weight given to each class. If given, has to be a Tensor of size C
   - size_average (bool, optional) – Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch. Note that for some losses, there multiple elements per sample. If the field size_average is set to False, the losses are instead summed for each minibatch. Ignored when reduce is False. Default: True
   - ignore_index (int, optional) – Specifies a target value that is ignored and does not contribute to the input gradient. When size_average is True, the loss is averaged over non-ignored targets. Note that ignore_index is only applicable when the target contains class indices. Default: -100
   - reduce (bool, optional) – Deprecated (see reduction). By default, the losses are averaged or summed over observations for each minibatch depending on size_average. When reduce is False, returns a loss per batch element instead and ignores size_average. Default: True
   - reduction (str, optional) – Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'
   - label_smoothing (float, optional) – A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution as described in Rethinking the Inception Architecture for Computer Vision. Default: 0.0.

    **Return type:**

   - Tensor

0. ## **torch.optim**

   > torch.optim is a package implementing various optimization algorithms.

   Construction:

   - To construct an Optimizer you have to give it an iterable containing the parameters (all should be Variable s) to optimize. Then, you can specify optimizer-specific options such as the learning rate, weight decay, etc.

    ```
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam([var1, var2], lr=0.0001)
    ```
