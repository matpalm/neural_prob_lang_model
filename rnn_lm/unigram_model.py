#!/usr/bin/env python

# unigram model
import sys
import numpy as np
from collections import defaultdict
from util import load_training_test, perplexities_and_second_last_probs

training, test = load_training_test(sys.argv[1], sys.argv[2])

# training; record frequencies per symbol, later used to determine P(w)
token_freq = defaultdict(int)
total = 0
for seq in training:
    for c in seq:
        token_freq[c] += 1
    total += len(seq)

# test; probability of anything is based on unigram frequency
prob_seqs = []
for seq in test:
    probs = [float(token_freq[c]) / total for c in seq]    
    prob_seqs.append(probs)
print perplexities_and_second_last_probs(prob_seqs)
