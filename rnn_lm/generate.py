#!/usr/bin/env python

# generate a sequence of embedded reber string
# see original LSTM paper, s5.1 exp1

import random, sys

def coin_flip():
    return 0 if random.random() < 0.5 else 1

def reber_sequence():
    transistions = { 0: [(1, "T"), (3, "P")],
                     1: [(1, "S"), (2, "X")],
                     2: [(3, "X"), (5, "S")],
                     3: [(3, "T"), (4, "V")],
                     4: [(2, "P"), (5, "V")] }
    state, s = 0, ["B"]
    while state != 5:
        transistion = transistions[state][coin_flip()]
        state = transistion[0]
        s.append(transistion[1])
    s.append("E")
    return s

def embedded_reber_sequence():
    path = "T" if coin_flip() else "P"
    return ["B"] + [path] + reber_sequence() + [path] + ["E"]


num_to_generate = None
try:
    num_to_generate = int(sys.argv[1])
except:
    raise Exception("usage: " + sys.argv[0] + " num_strings_to_generate")

for _ in xrange(0, num_to_generate):
    print "".join(embedded_reber_sequence())
