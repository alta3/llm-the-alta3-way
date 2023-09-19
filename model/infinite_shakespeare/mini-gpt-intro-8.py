for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input from tensor 1 is {context.tolist()} the target found in tensor 2 is: {target}")
