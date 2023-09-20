# On larger data sets with finer training required you'd want smaller gradient, 3e-4 (.00004 vs .0001)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(25): # increase number of steps for good results... 

    # sample a batch of training data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    # Resets the gradient's to 0 to begin each loop. This helps prevent the numbers for compounding on each other and     # either vanishing to 0 or exploding to incredibly high numbers.
    optimizer.zero_grad(set_to_none=True)
    # backpropogate through the network, computing the gradients of the model's parameters
    loss.backward()
    # AdamW optimizer applies the computed gradients to update the model's weights/parameters
    optimizer.step()
    # right now this prints EVERY loop
    print(loss.item())
