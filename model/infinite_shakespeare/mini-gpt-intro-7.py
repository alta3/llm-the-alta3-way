def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
  
    # first character/token stored in a stacked tensor
    x = torch.stack([data[i:i+block_size] for i in ix])
  
    # the second/next token from training, stored stocked in a second tensor 
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

# Now we'll print our variables to visualize what we've done
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)
