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

print("'Hii there' encoded looks like" , encode("hii there"))
print("Now we'll decode the phrase - " , decode(encode("hii there")))
print("You encoded" , decode(encode("hii there")) , "and it became" , encode("hii there"))
