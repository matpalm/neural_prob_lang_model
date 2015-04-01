#!/usr/bin/env python
# display convergence rate of a set of checkpoints over time

import numpy as np
import sys
import glob
import optparse
from collections import defaultdict
from checkpointer import prefix_time_var_of_ckpt

np.set_printoptions(linewidth=120)

optparser = optparse.OptionParser()
optparser.add_option('--files', None, dest='files', type='string', help='file glob')
optparser.add_option('--ids', None, dest='ids', type='string', default=None, help='comma seperated list of ids, if blank process all ids')
opts, arguments = optparser.parse_args()
print >>sys.stderr, "options", opts

ids = None
if opts.ids:
    ids = map(int, opts.ids.split(","))
print ids

if not opts.files: raise Exception("no --files specified?")
files = sorted(glob.glob(opts.files))
print files

# sanity check files are consistent with prefix and vars
prefix, _time, var = prefix_time_var_of_ckpt(files[0])
for f in files:
    next_prefix, _time, next_var = prefix_time_var_of_ckpt(f)
    if prefix != next_prefix or next_var != var: raise Exception("glob includes files that dont match in prefix or var")

# slurp in first
last = None
for f in files:
    X = np.load(f)
    if ids: X = X[ids]
    if last is not None:
        elemwise_diff_sqr = (X - last)**2
        row_sum = np.sum(elemwise_diff_sqr, axis=1)
        sqrt_row_sum = np.sqrt(row_sum)
        #print "X", X 
        #print "last", last
        #print "em_diff", elemwise_diff_sqr
        #print "row_sum", row_sum
        print "sqrt_row_sum", sqrt_row_sum
    last = X


exit(0)

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
