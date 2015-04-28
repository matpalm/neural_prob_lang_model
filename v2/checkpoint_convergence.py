#!/usr/bin/env python
# display convergence rate of a set of checkpoints over time

import numpy as np
import sys
import glob
import optparse
from collections import defaultdict
from checkpointer import prefix_time_var_of_ckpt
from token_idx import TokenIdx

np.set_printoptions(linewidth=120)

optparser = optparse.OptionParser()
optparser.add_option('--files', None, dest='files', type='string', help='file glob')
optparser.add_option('--tokens', None, dest='tokens', type='string', default=None, help='comma seperated list of tokens, if blank process all')
optparser.add_option('--vocab', None, dest='vocab', type='string', default=None, help='vocab file. token TAB id')
opts, arguments = optparser.parse_args()
print >>sys.stderr, "options", opts

if not opts.files:
    raise Exception("no --files specified?")
if "," in opts.files:
    files = opts.files.split(",")
else:
    files = sorted(glob.glob(opts.files))

token_idx = TokenIdx()
token_idx.read_from_file(opts.vocab)

# map tokens to ids
ids = None
if opts.tokens:
    ids = [token_idx.id_for(t) for t in opts.tokens.split(",")]

# sanity check files are consistent with prefix and vars
prefix, _time, var = prefix_time_var_of_ckpt(files[0])
for f in files:
    next_prefix, _time, next_var = prefix_time_var_of_ckpt(f)
    if prefix != next_prefix or next_var != var:
        raise Exception("glob includes files that dont match in prefix or var")

print "\t".join("ckpt_time idx token x_l_dist x_f_dist".split())

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
            print "%s\t%d\t%s\t%f\t%f" % (ckpt_time, t_id, token_idx.token_for(t_id), x_l_dist, x_f_dist)

    if first is None:
        first = X

    last = X
