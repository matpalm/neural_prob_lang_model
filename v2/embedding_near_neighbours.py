#!/usr/bin/env python
import numpy as np
from token_idx import TokenIdx
from sklearn.neighbors import LSHForest, BallTree
import optparse, sys

optparser = optparse.OptionParser(prog='embedding_near_neighbours.py', version='0.0.1', description='')
optparser.add_option('--vocab', None, dest='vocab', type='string', default='vocab.tsv', help='vocab for token idx')
optparser.add_option('--matrix-file', None, dest='matrix_file', type='string', help='np matrix file to load; eg ckpt.X.E')
optparser.add_option('--tokens', None, dest='tokens', type='string', help='space separated list of tokens to emit NNs for')
optparser.add_option('--k', None, dest='k', type='int', default=5, help='number of near neighbours to emit')
opts, arguments = optparser.parse_args()
print >>sys.stderr, "options", opts

token_idx = TokenIdx()
token_idx.read_from_file(opts.vocab)

# checking that tokens are in vocab
for token in opts.tokens.split(" "):
    if not token_idx.id_exists_for(token):
        print >>sys.stderr, "token [%s] not in vocab?" % token
        exit(1)

E = np.load(opts.matrix_file)

#lshf = LSHForest()
#lshf.fit(E)
#distances, indices = lshf.kneighbors(E[[token_idx.id_for("monday_NNS")]], n_neighbors=10)
#for d, i in zip(distances[0], indices[0]):
#    print d, token_idx.token_for(i)

ball_tree = BallTree(E, leaf_size=30)
for token in opts.tokens.split(" "):
    print
    distances, indices = ball_tree.query(E[[token_idx.id_for(token)]], k=min(opts.k, E.shape[0]))
    for d, nn in zip(distances[0], indices[0]):
        print d, token_idx.token_for(nn)
