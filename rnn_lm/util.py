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
