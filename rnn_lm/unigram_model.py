#!/usr/bin/env python

# unigram model
import sys
import numpy as np
from collections import defaultdict
from util import load_training_test, perplexity_of_sequence

training, test = load_training_test(sys.argv[1], sys.argv[2])

# training; record frequencies per symbol, later used to determine P(w)
token_freq = defaultdict(int)
total = 0
for seq in training:
    for c in seq:
        token_freq[c] += 1
    total += len(seq)

# test; probability of anything is based on unigram frequency
perplexities = []
for seq in test:
    seq_probabilities = [float(token_freq[c]) / total for c in seq]    
    perplexities.append(perplexity_of_sequence(seq_probabilities))
print "min, mean, max perplexity", min(perplexities), np.mean(perplexities), max(perplexities)
