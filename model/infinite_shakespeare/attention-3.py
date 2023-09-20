import torch
import torch.nn as nn
from torch.nn import functional as F

B,T,C = 4,8,32
# When we fold this into the model itself the x will become our embedding table of values for the token.
x = torch.randn(B,T,C)

# Add this entire section:
head_size = 16

# key can be viewed as the information the token contains about itself
key = nn.Linear(C, head_size, bias=False)

# query is essentially what is the token looking for? What does it pair well with?
query = nn.Linear(C, head_size, bias=False)

# value is the dot product of K and Q. In summary: Here's what the token has(k), here's what the token is interested in(q). 
# If you are aligned with me you will be looking for my value (v) 
# v is the FINAL aggregate score per attention head
value = nn.Linear(C, head_size, bias=False)

k = key(x)   # (B, T, 16) (head size controls the dimensional size of the B by T tensor)
q = query(x) # (B, T, 16)
# scaled dot product. Necessary to avoid the vanishing gradient problem. If you softmax wei without the scaled dot product the probabilities 
# will "sharpen" which will reduce variability. Tokens now have personal weights. (B, T, 16) @ (B, 16, T) ---> (B, T, T)

wei =  q @ k.transpose(-2, -1) * head_size**-0.5
tril = torch.tril(torch.ones(T, T))

#DELETE the  wei = torch.zeros((T,T)) line here, we are no longer creating a tensor of 0's
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

#Add this print line so we can see our weights
print("The generated (random) weights" , wei[0])

#Update the final output to include V
v = value(x)
out = wei @ v
#out = wei @ x

out.shape
