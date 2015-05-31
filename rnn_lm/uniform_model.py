#!/usr/bin/env python

# trivial uniform model
# expect perplexity == |vocab|

import sys
import numpy as np
from util import load_training_test, perplexities_and_second_last_probs

training, test = load_training_test(sys.argv[1], sys.argv[2])

# training; just record set of observed symbols
observed_symbols = set()
for seq in training:
    observed_symbols.update(seq)

# test; probability of anything in sequence is uniform
prob_seqs = []
uniform_prob = 1.0 / len(observed_symbols)
for seq in test:
    prob_seqs.append([uniform_prob] * len(seq))
print perplexities_and_second_last_probs(prob_seqs)
