#!/usr/bin/env python
import numpy as np
import sys, re
from checkpointer import prefix_time_var_of_ckpt

if len(sys.argv) != 2:
    print >>sys.stderr, "usage:", sys.argv[0], " <checkpoint_file_to_dump>"
    exit(1)
ckpt_file = sys.argv[1]

file_info = prefix_time_var_of_ckpt(ckpt_file)

for n, row in enumerate(np.load(ckpt_file)):
    print "\t".join(list(file_info) + [str(n)] + map(str, row))


