#!/usr/bin/env python
# generate some negative data for logistic loss training (as opposed to a softmax)
import sys, random
from collections import defaultdict, Counter

# slurp everything in
known_tokens = set()                # all known tokens
w0w1_w2freq = defaultdict(Counter)  # frequencies of w2s for a specific (w0, w1)
for line in sys.stdin:
    line = line.strip()
    print line, "1"
    w = line.split(" ")
    if len(w) != 3:
        print >>sys.stderr, "expected 3 tokens; not [", line, "]"
        exit(1)
    known_tokens.update(w)
    w0w1_w2freq[w[0] + " " + w[1]][w[2]] += 1

for w0w1, w2_freq in w0w1_w2freq.iteritems():
    negative_candidates = list(known_tokens - set(w2_freq.keys()))
    num_positives = sum(w2_freq.values())
    for i in range(num_positives):
        print w0w1, random.choice(negative_candidates), "0"


