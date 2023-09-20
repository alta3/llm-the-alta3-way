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

print(logits.shape)
print(loss)

# IDX is a 1,1 tensor that starts with 0. This is the 'input' or the 'first token' that the generator will use to launch itself. 
# The 0 index maps to the 'new line character' in our models vocabulary.
idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
