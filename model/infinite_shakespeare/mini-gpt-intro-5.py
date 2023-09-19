# Set the first 90% of our dataset for training. 
# The remainder will be used for validation.
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
