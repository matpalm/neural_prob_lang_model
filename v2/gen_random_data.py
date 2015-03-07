#!/usr/bin/env python
import networkx as nx
from collections import defaultdict
import random as r
import optparse
import sys

optparser = optparse.OptionParser(prog='nplp', version='0.0.1', description='simple neural probabilistic language model')
optparser.add_option('--seed', None, dest='seed', type='int', default=None, help='rng seed')
optparser.add_option('--to-dot', action="store_true", dest='to_dot', help='dump dot format and exit')
optparser.add_option('--er-n', None, dest='er_n', type='int', default=10, help='erdos_renyi n param')
optparser.add_option('--er-p', None, dest='er_p', type='float', default=0.15, help='erdos_renyi p param')
optparser.add_option('--generate', None, dest='gen', type='int', default=10, help='number of sequences to generate')
opts, arguments = optparser.parse_args()

if opts.seed is not None:
    r.seed(int(opts.seed))

er = nx.erdos_renyi_graph(opts.er_n, opts.er_p, directed=True)

labels = ['A', 'B', 'C', 'D', 'E', 'F']
def label(i):
    return labels[i % len(labels)]

if opts.to_dot:
    print "digraph G { rankdir=LR; "
    for i, j in er.edges():
        print "%s_%s -> %s_%s" % (label(i), i, label(j), j)
    print "}"
    exit(0)

print >>sys.stderr, "options", opts

adj = defaultdict(list)
for i, j in er.edges():
    adj[i].append(j)

t = defaultdict(list)
for node, adj_nodes in adj.iteritems():
    proportion = 1.0 / len(adj_nodes)
    for i, adj_node in enumerate(adj_nodes):
        t[node].append((adj_node, (i+1) * proportion))

#for node, adj_nodes in t.iteritems():
#    print node, adj_nodes
#print "keys", t.keys()

def next_from(n):
    rnd = r.random()
    i = 0
    while rnd >= t[n][i][1]:
        i += 1
    return t[n][i][0]

generated = 0 
bail = 100
while generated < opts.gen:
    n = r.choice(t.keys())
    seq = [n]
    while n in t.keys():  # ie has neighbours
        n = next_from(n)
        seq.append(n)
    print " ".join([label(i) for i in seq])
    generated += 1
        
    



