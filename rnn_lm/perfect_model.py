#!/usr/bin/env python

# trivial (testing) perfect model; predicts sequence perfectly
# expect perplexity == 1.0

import sys
import numpy as np
from util import load_training_test, perplexity_of_sequence

_training, test = load_training_test(sys.argv[1], sys.argv[2])

# training; just record set of observed symbols
# ignore data

# test; assume perfect prediction
perplexities = []
for seq in test:
    perplexities.append(perplexity_of_sequence([1.0] * len(seq)))
print "min, mean, max perplexity", min(perplexities), np.mean(perplexities), max(perplexities)
