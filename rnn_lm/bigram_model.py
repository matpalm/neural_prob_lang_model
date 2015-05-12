#!/usr/bin/env python

# bigram model

import sys
import numpy as np
from collections import defaultdict
from util import load_training_test, perplexity_of_sequence

training, test = load_training_test(sys.argv[1], sys.argv[2])

# training; calculate bigram probabilities P(w2|w1)
bigram_freq = defaultdict(lambda: defaultdict(int))
for seq in training:
    for i in xrange(1, len(seq)):
        bigram_freq[seq[i-1]][seq[i]] += 1
bigram_prob = {}
for w1 in bigram_freq.keys():
    bigram_prob[w1] = {}
    w2_freqs = bigram_freq[w1]
    w2_total = float(sum(w2_freqs.values()))
    for w2, freq in w2_freqs.iteritems():
        bigram_prob[w1][w2] = float(bigram_freq[w1][w2]) / w2_total

# test
perplexities = []
for seq in test:
    # small vocab so assume all bigrams have data from training (ie no need to backoff to unigram model)
    bigram_probabilities = [bigram_prob[seq[i-1]][seq[i]] for i in xrange(1, len(seq))]
    perplexities.append(perplexity_of_sequence(bigram_probabilities))
print "min, mean, max perplexity", min(perplexities), np.mean(perplexities), max(perplexities)