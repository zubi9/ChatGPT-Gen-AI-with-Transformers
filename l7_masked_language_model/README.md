# Masked Language Model

- The main script in this directory trains an encoder-only transformer as a masked language model.
- The dataset `df_sequences.json` is a dictionary of event log sequences.
- Each key is a case id, and its value is a sequence of activities.
- Each activity is considered to be one token. 
- For example ["Assign seriousness", "Take in charge ticket", "Take in charge ticket", "Resolve ticket", "Closed"] should be tokenized into something like [4, 6, 6, 7, 15] 
- Tokens like `<sos>` (start of sentence) and `<eos>` (end of sentence) should also be added.
- In this example, the tokenized input would become [1, 4, 6, 6, 7, 15, 2] with tokens 1 and 2 representing the respective special tokens.
- The tokenized sequence is to be masked with masking token (for example 3) and with a certain probability.
- Note that the special (start and end) tokens are not to be masked.
- Therefore, from an activity sequence, you derive two sequences: one is tokenized (target), and one is the masked version of the tokenized (input)
- You train your encoder as a masked language model to demask the input. However, you need to pad all of your data into a fixed length (for example, you add 0s to make them all 32D).
- After training, the MLM head is detached (encoder takes `x` as output layer, which is the output of the transformer block), and the trained encoder is ready to embed sequences from their tokenized form into their vectors in the latent space.


Finish the functions in `my_utils.py` and the last part of `main.py` to perform these points. 