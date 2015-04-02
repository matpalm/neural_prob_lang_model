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

if not opts.files:
    raise Exception("no --files specified?")
if "," in opts.files:
    files = opts.files.split(",")
else:
    files = sorted(glob.glob(opts.files))

# sanity check files are consistent with prefix and vars
prefix, _time, var = prefix_time_var_of_ckpt(files[0])
for f in files:
    next_prefix, _time, next_var = prefix_time_var_of_ckpt(f)
    if prefix != next_prefix or next_var != var:
        raise Exception("glob includes files that dont match in prefix or var")

first = None  # always compare to first
last = None   # also compare to last
for f in files:
    X = np.load(f)

    if ids:
        X = X[ids]

    if last is not None:
        _prefix, ckpt_time, _var = prefix_time_var_of_ckpt(f)
        row_wise_distance_X_L = np.sqrt(np.sum((X - last) ** 2, axis=1))
        row_wise_distance_X_F = np.sqrt(np.sum((X - first) ** 2, axis=1))
        for i, (x_l_dist, x_f_dist) in enumerate(zip(row_wise_distance_X_L, row_wise_distance_X_F)):
            t_id = ids[i] if ids else i
            print "%s\t%d\t%f\t%f" % (ckpt_time, t_id, x_l_dist, x_f_dist)

    if first is None:
        first = X

    last = X
