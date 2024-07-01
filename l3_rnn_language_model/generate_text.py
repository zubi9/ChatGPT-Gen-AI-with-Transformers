import argparse
import pickle
import numpy as np
from train import RNNGenerator

def load_model(filepath):
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def main():
    parser = argparse.ArgumentParser(description='Generate text using a trained RNN model.')
    parser.add_argument('model_file', type=str, help='Path to the saved model file.')
    parser.add_argument('seed_text', type=str, help='Seed text to start generation.')
    parser.add_argument('--length', type=int, default=1000, help='Length of the generated text (default: 1000).')
    args = parser.parse_args()

    # Load the trained model
    model_data = load_model(args.model_file)

    # Initialize the generator with loaded model parameters
    generator = RNNGenerator(
        textfile='ai_faq,  # We do not need to specify a textfile here
        enc='utf-8',  # This can be any encoding; it won't be used
        mode='chars',  # Use characters mode
        hidden_size=model_data['hidden_size'],
        seq_length=model_data['seq_length'],
        learning_rate=model_data['learning_rate']
    )

    # Set the model weights
    generator.Wxh = model_data['Wxh']
    generator.Whh = model_data['Whh']
    generator.Why = model_data['Why']
    generator.bh = model_data['bh']
    generator.by = model_data['by']
    generator.token_to_ix = model_data['token_to_ix']
    generator.ix_to_token = model_data['ix_to_token']
    generator.h = model_data['h']
    generator.vocab_size = len(generator.token_to_ix)

    # Encode the seed text
    seed_ix = [generator.token_to_ix[char] for char in args.seed_text]
    generated_text = []
    
    # Generate text using the seed
    for ix in seed_ix:
        generated_text.append(generator.ix_to_token[ix])
    generated_text += generator.sample(seed_ix[-1], args.length - len(seed_ix))

    # Print the generated text
    print("".join(generated_text))

if __name__ == "__main__":
    main()
