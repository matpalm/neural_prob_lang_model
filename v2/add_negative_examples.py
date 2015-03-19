#!/usr/bin/env python
# generate some negative data for logistic loss training (as opposed to a softmax)
# emit uniformally randomly selected negatives (unobserved) 

import sys, random, itertools
from collections import defaultdict, Counter

# slurp everything in
observed_w = set()     # all observed w
observed_triples = set()  # all observed w1w2w3 tuples
num_observations = 0
for line in sys.stdin:
    line = line.strip()
    w = line.split(" ")
    if len(w) != 3:
        print >>sys.stderr, "expected 3 tokens; not [", line, "]"
        exit(1)

    print line, "1"

    observed_w.update(w)
    observed_triples.add(tuple(w))
    num_observations += 1

# decide which triples were not observed
all_triples = set(itertools.product(observed_w, repeat=3))
unobserved_triples = all_triples - observed_triples
unobserved_triples = [ " ".join(t) for t in unobserved_triples ]

for _i in range(num_observations):
    print random.choice(unobserved_triples), "0"

#for w0w1, w2_freq in w0w1_w2freq.iteritems():
#    negative_candidates = list(known_tokens - set(w2_freq.keys()))
#    num_positives = sum(w2_freq.values())
#    for i in range(num_positives):
#        print w0w1, random.choice(negative_candidates), "0"


