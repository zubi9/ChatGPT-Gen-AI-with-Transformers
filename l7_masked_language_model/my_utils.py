import numpy as np
from sklearn.model_selection import train_test_split

def get_vocab_size(df_sequences):
    # write code to get vocab_size from the sequences in df_sequences
    # vocab_size is the size of vocabulary, i.e., how many distinct tokens there are.

    # hint: fill the following dictionary of tokens from your data
    df_tokens = {
        "<pad>": 0,
        "<sos>": 1,
        "<eos>": 2,
        "<mask>": 3,
    }
    activity_set = set()
    for key in df_sequences.keys():
        activity_set.update(df_sequences[key])
    token = 4
    for activity in activity_set:
        df_tokens[activity] = token
        token +=1
    vocab_size = len(df_tokens.keys())
    return vocab_size, df_tokens


def get_tokenized_inputs(df_sequences, df_tokens):
    df_tokenized_inputs = {}
    # For every sequence in df_sequences, fill df_tokenized_inputs with corresponding tokens
    for key in df_sequences.keys():
        sequence = df_sequences[key]
        tokenized_input = [df_tokens["<sos>"]]
        for word in sequence:
            tokenized_input.append(df_tokens[word])
        tokenized_input.append(df_tokens["<eos>"])
        df_tokenized_inputs[key] = tokenized_input
    return df_tokenized_inputs

def get_masked_tokenized_inputs(df_tokenized_inputs, masking_probability, df_tokens):
    df_masked_tokenized_inputs = {}

    # For every tokenized sequence in df_tokenized_inputs, mask it with masking_probability to fill df_masked_tokenized_inputs
    for key in df_tokenized_inputs.keys():
        sequence = df_tokenized_inputs[key][1:][:-1]
        mask = np.random.rand(len(sequence)) < (masking_probability)
        sequence_masked = np.where(mask, df_tokens["<mask>"], sequence).tolist()
        sequence_masked = [df_tokens["<sos>"]]+sequence_masked+[df_tokens["<eos>"]]
        df_masked_tokenized_inputs[key] = sequence_masked
    return df_masked_tokenized_inputs


def pad_processed_data(maxlen, df_tokens, df_tokenized_inputs, df_masked_tokenized_inputs):
    df_tokenized_inputs_padded = {}
    df_masked_tokenized_inputs_padded = {}


    # Pad every item in the tokenized and masked dictionaries
    # Hint: start with <sos> and end with <eos> and then fill the rest with <pad>
    # At the end, all items in both dictionaries must be of length maxlen

    for key in df_tokenized_inputs.keys():
        sequence = df_tokenized_inputs[key]
        if len(sequence) < maxlen:
            sequence = sequence+ [df_tokens["<pad>"]]*(maxlen-len(sequence))
        df_tokenized_inputs_padded[key] = sequence

        sequence = df_masked_tokenized_inputs[key]
        if len(sequence) < maxlen:
            sequence = sequence+ [df_tokens["<pad>"]]*(maxlen-len(sequence))
        df_masked_tokenized_inputs_padded[key] = sequence

    return df_tokenized_inputs_padded, df_masked_tokenized_inputs_padded


def get_train_test_split(df_tokenized_inputs_padded, df_masked_tokenized_inputs_padded):
    case_ids = list(df_tokenized_inputs_padded.keys())
    
    train_split, temp_split, _, _ = train_test_split(case_ids, case_ids, test_size=0.3)
    test_split, unseen_split, _, _ = train_test_split(temp_split, temp_split, test_size=0.3)    
    df_split = {
        "train_split": train_split,
        "test_split": test_split,
        "unseen_split": unseen_split
    }        

    df_train_test = {
        "train_input":[], "train_masked":[],
        "test_input":[], "test_masked":[]
    }
    for key in ["train", "test"]:
        for case_id in df_split[key+"_split"]:
            df_train_test[key+"_input"].append(df_tokenized_inputs_padded[case_id])
            df_train_test[key+"_masked"].append(df_masked_tokenized_inputs_padded[case_id])
    
    return df_train_test


