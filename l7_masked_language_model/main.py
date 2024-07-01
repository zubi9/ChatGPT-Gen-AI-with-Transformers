import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")



import json, numpy as np
from architecture import *
from my_utils import *

# TASK: finish the functions in my_utils.py so that main.py can run

with open("data/sequences.json") as f:
    df_sequences = json.load(f)

maxlen = 32

vocab_size, df_tokens = get_vocab_size(df_sequences)

df_tokenized_inputs = get_tokenized_inputs(df_sequences, df_tokens)

masking_probability = 0.2  # 20% 
df_masked_tokenized_inputs = get_masked_tokenized_inputs(df_tokenized_inputs, masking_probability, df_tokens)

df_tokenized_inputs_padded, df_masked_tokenized_inputs_padded = pad_processed_data(maxlen, df_tokens, df_tokenized_inputs, df_masked_tokenized_inputs)

df_train_test = get_train_test_split(df_tokenized_inputs_padded, df_masked_tokenized_inputs_padded)

embed_dim = maxlen
ff_dim = 128
num_heads = 4

model, x = get_masked_language_model(maxlen, vocab_size, embed_dim, num_heads, ff_dim)

epochs = 10
batch_size = 12
model.fit(np.array(df_train_test["train_masked"]),np.array(df_train_test["train_input"]), epochs=epochs, batch_size=batch_size, 
          validation_data=(np.array(df_train_test["test_masked"]), np.array(df_train_test["test_input"])))


encoder = Model(inputs=model.input, outputs=x)

df_embeddings = {}
for key in df_tokenized_inputs_padded.keys():
    input_data = df_tokenized_inputs_padded[key]
    input_data = tf.expand_dims(input_data, 0)
    encoded_vector = encoder.predict(input_data)
    avg_output = np.mean(encoded_vector, axis=1)
    df_embeddings[key] = avg_output[0].tolist()
with open("data/embeddings.json", "w") as f:
    json.dump(df_embeddings, f)

# run after interactive_map.py after storing the embeddings