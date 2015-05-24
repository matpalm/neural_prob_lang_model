#!/usr/bin/env python

# generate a sequence of embedded reber string
# see original LSTM paper, s5.1 exp1

import random, sys, optparse

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

optparser = optparse.OptionParser(prog='generate', description='generate embedded reber strings')
optparser.add_option('--num', None, dest='num', type='int', default=10,
                     help='number of strings to generate')
optparser.add_option('--seq-length', None, dest='seq_length', type='int', default=None,
                     help='if set, only generate of this length. must be >=10')
opts, arguments = optparser.parse_args()

if opts.seq_length != None and opts.seq_length < 10:
    raise Exception("seq-length must be >=10")

for _ in xrange(0, opts.num):
    while True:  # o_O
        candidate = embedded_reber_sequence()
        if opts.seq_length == None or len(candidate) == opts.seq_length:
            print "".join(candidate)
            break

