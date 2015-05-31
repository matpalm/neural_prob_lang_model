#!/usr/bin/env python

# trivial (testing) perfect model; predicts sequence perfectly
# expect perplexity == 1.0

import sys
import numpy as np
from util import load_training_test, perplexities_and_second_last_probs

_training, test = load_training_test(sys.argv[1], sys.argv[2])

# training; just record set of observed symbols
# ignore data

# test; assume perfect prediction
prob_seqs = []
for seq in test:
    prob_seqs.append([1.0] * len(seq))
print perplexities_and_second_last_probs(prob_seqs)
