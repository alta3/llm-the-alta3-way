import torch

torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)

weight = torch.tril(torch.ones(T,T))
weight = weight / weight.sum(1, keepdim = True)
print(weight)
