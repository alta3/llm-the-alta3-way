# import pytorch
import torch
import torch.nn as nn
from torch.nn import functional as F

# Open the file for reading 
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# here are all the unique characters that occur in this text --This provides the library of all possible Tokens
# The following code uses set() to extract all UNIQUE characters from the input.txt, list() puts those 
# characters into a single list, and sorted() sorts the list.

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Unlike with TikToken, we're going to us lambda for our tokenization
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)

# Set the first 90% of our dataset for training.
# The remainder will be used for validation.
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# the manual seed enforces order upon our random data sampling so that we can measure expected results
torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # first character/token stored in a stacked tensor
    x = torch.stack([data[i:i+block_size] for i in ix])

    # the second/next token from training, stored stocked in a second tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

class MiniGPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
        # Here we must mutate the tensor slightly in order to align with pyTorch module expectations.
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the predictions (calls def forward)
            logits, loss = self(idx)
            # focus only on the last token created, essentially locking in the T and shaping the tensor to B and C.
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution (take a single sample)
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the current running sequence, folding time back in and creating B, T+1
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = MiniGPTLanguageModel(vocab_size)
# calls the model with tensor of inputs and outputs generated in data chunking step
logits, loss = m(xb, yb)

# On larger data sets with finer training required you'd want smaller gradient, 3e-4 (.00004 vs .0001)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(3000): # increase number of steps for good results... 

    # sample a batch of training data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    # Resets the gradient's to 0 to begin each loop. This helps prevent the numbers for compounding on each other and 
    # either vanishing to 0 or exploding to incredibly high numbers.
    optimizer.zero_grad(set_to_none=True)
    # backpropogate through the network, computing the gradients of the model's parameters
    loss.backward()
    # AdamW optimizer applies the computed gradients to update the model's weights/parameters
    optimizer.step()
    # right now this prints EVERY loop
    print(loss.item())

# IDX is a 1,1 tensor that starts with 0. This is the 'input' or the 'first token' that the generator will use to launch itself. 
# The 0 index maps to the 'new line character' in our models vocabulary.
idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
