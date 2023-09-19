# here are all the unique characters that occur in this text --This provides the library of all possible Tokens
# The following code uses set() to extract all UNIQUE characters from the input.txt, list() puts those

characters into a single list, and sorted() sorts the list. 
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print("Input contains" , vocab_size , "possible tokens")
