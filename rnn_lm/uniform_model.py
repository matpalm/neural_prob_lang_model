#!/usr/bin/env python

# trivial uniform model
# expect perplexity == |vocab|

import sys
import numpy as np
from util import load_training_test, perplexity_of_sequence

training, test = load_training_test(sys.argv[1], sys.argv[2])

# training; just record set of observed symbols
observed_symbols = set()
for seq in training:
    observed_symbols.update(seq)

# test; probability of anything in sequence is uniform
perplexities = []
uniform_prob = 1.0 / len(observed_symbols)
for seq in test:
    perplexities.append(perplexity_of_sequence([uniform_prob] * len(seq)))
print "min, mean, max perplexity", min(perplexities), np.mean(perplexities), max(perplexities)
