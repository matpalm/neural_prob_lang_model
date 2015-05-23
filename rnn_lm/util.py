import sys
import numpy as np

def read_sequences(filename):
    lines = map(str.strip, open(filename).readlines())
    sequences = [[c for c in line] for line in lines]
    return sequences

def load_training_test(training_file, test_file):
    return (read_sequences(training_file), read_sequences(test_file))

def perplexity_of_sequence(probabilities):
    perplexity_sum = sum([np.log2(max(1e-10, p)) for p in probabilities])
    return 2**((-1./len(probabilities)) * perplexity_sum)

class TokenIdx(object):  # todo: is there nothing like this in scipy/numpy ? hmm :/

    def __init__(self):
        self.token_idx = {}
        self.idx_token = {}
        self.seq = 0

    def id_exists_for(self, token):
        return token in self.token_idx

    def id_for(self, token):
        if token in self.token_idx:
            return self.token_idx[token]
        else:
            self.token_idx[token] = self.seq
            self.idx_token[self.seq] = token
            self.seq += 1
            return self.seq - 1

    def ids_for(self, tokens):
        return [self.id_for(t) for t in tokens]

    def token_for(self, idx):
        return self.idx_token[idx]

    def labels(self):
        return [self.idx_token[i] for i in range(self.seq)]

    def num_entries(self):
        return self.seq


