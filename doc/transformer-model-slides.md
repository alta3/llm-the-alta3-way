# Slide 1 - Embeddings

**Tensor** - A Data Structure representing a multidimensional array of scalar values, vectors, and matrices. 
**Vector** - A numerical representation capturing the meaning of a word or token.
**Embeddings** - Dense Vector (multiple vectors) representations of tokens which capture semantic information about the tokens, meaning that similar words or characters will have similar vector representations.

The key/value/query concept is analogous to retrieval systems. For example, when you search for videos on Youtube, the search engine will map your query (text in the search bar) against a set of keys (video title, description, etc.) associated with candidate videos in their database, then present you the best matched videos (values).

The queries are the prompt text, the keys are the parameters (pre-trained language model weights, embeddings, or additional constraints provided to the model), and the values are the computed result that needs to be decoded.

**Query** - The Prompt Text (What We sent the Model)
**Key** - The Parameters (Pre-Trained Language Model Weights, Embeddings, or Additional Constraints)
**Value** - Computed Results from the Encoder/Decdoder

# Slide 2 - Encoder and Decoder

Figure of the Encoder and Decoder Stacks

The encoder and decoder stacks are a series of identical layers that are stacked together to form the overall model.

Encoder Stack: (Turns Words to Numbers) Takes the output of the previous layer and applies a series of operations to it, such as attention and feedforward networks. The output of the final layer in the encoder stack is then passed to the decoder stack.

Decoder Stack: (Turns Numbers to Words) The decoder stack attends to the previous words in the output sequence (Output Embeddings). This is because the decoder is trying to generate the next word in the sequence.

Mask: Will prevent the Decoder from attempting to predict the future at that point. This occurs only at the first multi-head attention process in the Decoder.

# Slide 3 - Multi-Head Attention (Scaled Dot-Product Attention)

Figure of Multi-Head Attention (L)

Multi-Head Attention allows the model to focus on different parts of the input sequence simultaneously, allowing it to capture complex relationships and dependencies.

It does this using a mechanism called Scaled Dot-Pattern Attention

Figure of Scaled Dot-Product Attention (R)

The mechanism by which a query is compared to the keys to generate a value.

Matrix Multiplexer used to compute the attention weights in the attention mechanism. Attention Weight is the vector which informs the model of its semantic connections. 

Softmax layer outputs a probability distribution over the vocabulary, which is used to select the next word in the output sequence.

# Slide 4 - Add & Norm

The technique used to improve stability and performance of the model. The output of each layer is added to the input of the next layer. The Sum is then normalized to ensure the outputs are within a specific range.

This helps to make the output of the layer more independent of the input and helps to improve the stability of the model.

Effective in preventing overfitting and improving performance of the model.

Overfitting - Overfitting is a machine learning problem that occurs when a model learns the training data too well and is unable to generalize to new data.

# Slide 5 - Feed Forward Nueral Network

The weights calculated by the Multi-Head Attention Mechanisms (Values), will be used to predict the next token.

Figure of the Feed Forward Neural Netork

# Slide 6 - Output Embeddings

Output Embeddings: Are the output of the final layer of the decoder. They are a sequence of vectors that represent the predicted words in the output sequence. Included in the output embeddings are the attention weights that were used to generate the output embeddings. 
