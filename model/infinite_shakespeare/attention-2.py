import torch
import torch.nn as nn
from torch.nn import functional as F

B,T,C = 4,8,32 # Bumped channels to 32 to match our MiniGPT model for ease later.
x = torch.randn(B,T,C)

tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
# turn the 0's into negative infinity, softmax will turn them back to 0
wei = wei.masked_fill(tril == 0, float('-inf'))
# turns all values into probabilities - 
wei = F.softmax(wei, dim=-1)
out = wei @ x
out.shape
