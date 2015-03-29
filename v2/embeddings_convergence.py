#!/usr/bin/env python
# display convergence rate of an embeddings (ie eucledian distance between subsequent embeddings dumps)

import numpy as np
import sys
import optparse
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option('--tokens', None, dest='tokens', type='string', help='tokens, comma seperated, to dump')
opts, arguments = optparser.parse_args()
tokens_to_keep = opts.tokens.split(",")

id_cols = defaultdict(list)
d_n_points = defaultdict(list)
d_n = None
for line in sys.stdin:
    cols = line.strip().split("\t")
    token = cols[3]
    if not token in tokens_to_keep:
        continue

    id_cols[token].append(cols[0:4])    
    d_n_points[token].append(np.asarray(map(float, cols[4:])))

    if d_n == None:
        d_n = len(cols) - 4
    elif d_n != len(cols) - 4:
        raise Exception("first # data points was %s but not it's %s ??" % (d_n, (len(cols)-4)))

# TODO: angles on unit sphere for normalised case

print "epoch\tbatch\tidx\tlabel\tl2_distance"

for token in id_cols.keys():
    last_data_point = None
    for id_col, d_n_point in zip(id_cols[token], d_n_points[token]):
        if last_data_point is None:
            # collect first row
            last_data_point = d_n_point
        else:
            # emit distance c.f. last row
            print "%s\t%s" % ("\t".join(id_col), np.linalg.norm(d_n_point - last_data_point))
            last_data_point = d_n_point
