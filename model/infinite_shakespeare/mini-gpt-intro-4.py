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

# Utilize PyTorch to transform our encoded data into the Tensor Data Structure
data = torch.tensor(encode(text), dtype=torch.long)
print(data[:1000]) # the 1000 characters we looked at earier will now be in a tensor to compare
print(data.shape, data.dtype)
