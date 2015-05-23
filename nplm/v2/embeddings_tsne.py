#!/usr/bin/env python
# rewrite an n-dimensional embedding dump as 2d via TSNE

from sklearn.manifold import TSNE
import numpy as np
import sys

id_cols = []
d_n_points = []
for line in sys.stdin:
    cols = line.strip().split("\t")
    id_cols.append(cols[0:4])
    d_n_points.append(map(float, cols[4:]))

model = TSNE(n_components=2, random_state=0)
d_2_points = model.fit_transform(d_n_points)

sys.stdout.write("epoch\tbatch\tidx\tlabel\td0\td1\n")
for id_col, d_2_point in zip(id_cols, d_2_points):
    sys.stdout.write("\t".join(id_col))
    sys.stdout.write("\t")
    sys.stdout.write("\t".join(map(str, d_2_point)))
    sys.stdout.write("\n")
