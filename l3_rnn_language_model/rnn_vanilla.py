import argparse
import pickle
import numpy as np

def tokens_from_characters(textfile, enc):
    data = open(textfile, 'r', encoding=enc).read()
    tokens = list(set(data))
    return tokens, data

def tokens_from_words(textfile, enc):
    data = open(textfile, 'r', encoding=enc).read()
    data = data.split()
    tokens = list(set(data))
    return tokens, data

def vectors_from_tokens(tokens):
    token_to_ix = {t: i for i, t in enumerate(tokens)}
    ix_to_token = {i: t for i, t in enumerate(tokens)}
    return token_to_ix, ix_to_token

class RNNGenerator:
    def __init__(self, textfile, enc='utf-8', mode="words", hidden_size=100, seq_length=25, learning_rate=1e-1):
        self.mode = mode
        self.tokens, self.data = tokens_from_words(textfile, enc=enc) if self.mode == "words" else tokens_from_characters(textfile, enc=enc)
        self.token_to_ix, self.ix_to_token = vectors_from_tokens(self.tokens)
        self.vocab_size = len(self.tokens)
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        # Initialize model parameters
        self.Wxh = np.random.randn(hidden_size, self.vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(self.vocab_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))

        self.h = np.zeros((hidden_size, 1))  # Initialize hidden state
        self.p = 0  # Initialize position in data
        self.n = 0  # Initialize iteration count

        self.smooth_loss = -np.log(1.0 / self.vocab_size) * seq_length
        self.loss_history = []
    def lossFun(self, inputs, targets):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(self.h)
        loss = 0
        # forward pass
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))  # encode in 1-of-k representation
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)  # hidden state
            ys[t] = np.dot(self.Why, hs[t]) + self.by  # unnormalized log probabilities for next words
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next words
            loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # backprop into y
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext  # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
        self.h = hs[len(inputs) - 1]
        return loss, dWxh, dWhh, dWhy, dbh, dby

    def sample(self, seed_ix, n):
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = [seed_ix]
        h = self.h
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return [self.ix_to_token[ix] for ix in ixes]

    def _reset_to_start(self):
        self.p = 0
        self.h = np.zeros((self.hidden_size, 1))

    def step(self):
        if self.p + self.seq_length + 1 >= len(self.data) or self.n == 0:
            self._reset_to_start()
        inputs = [self.token_to_ix[word] for word in self.data[self.p:self.p + self.seq_length]]
        targets = [self.token_to_ix[word] for word in self.data[self.p + 1:self.p + self.seq_length + 1]]

        loss, dWxh, dWhh, dWhy, dbh, dby = self.lossFun(inputs, targets)
        self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
        self.loss_history.append(self.smooth_loss)  # Store the smooth loss

        for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
            mem += dparam * dparam
            param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update
        self.p += self.seq_length
        self.n += 1

    def save_model(self, filepath):
        model_data = {
            'Wxh': self.Wxh,
            'Whh': self.Whh,
            'Why': self.Why,
            'bh': self.bh,
            'by': self.by,
            'token_to_ix': self.token_to_ix,
            'ix_to_token': self.ix_to_token,
            'hidden_size': self.hidden_size,
            'seq_length': self.seq_length,
            'learning_rate': self.learning_rate,
            'smooth_loss': self.smooth_loss,
            'h': self.h
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f'Model saved to {filepath}')

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.Wxh = model_data['Wxh']
        self.Whh = model_data['Whh']
        self.Why = model_data['Why']
        self.bh = model_data['bh']
        self.by = model_data['by']
        self.token_to_ix = model_data['token_to_ix']
        self.ix_to_token = model_data['ix_to_token']
        self.hidden_size = model_data['hidden_size']
        self.seq_length = model_data['seq_length']
        self.learning_rate = model_data['learning_rate']
        self.smooth_loss = model_data['smooth_loss']
        self.h = model_data['h']
        self.vocab_size = len(self.token_to_ix)  # Update vocab_size
        print(f'Model loaded from {filepath}')
