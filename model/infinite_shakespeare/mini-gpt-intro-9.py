# import pytorch
import torch

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
