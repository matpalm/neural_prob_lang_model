from __future__ import print_function
import sys
import numpy as np
from token_idx import TokenIdx

def load_trigram_data(filename, mode):
    token_idx = TokenIdx()
    idxs = [] 
    target = []
    for n, line in enumerate(open(filename, 'r')):
        if n%100000==0: sys.stderr.write("loading data %s\r" % n)
        record = line.strip().split()
        w1_idx, w2_idx, w3_idx = [token_idx.id_for(w) for w in record[:3]]
        if len(record) == 3 and mode=='sm':
            # just trigram
            idxs.append([w1_idx, w2_idx])
            target.append(w3_idx)
        elif len(record) == 4 and mode=='lr':
            # trigram with label
            idxs.append([w1_idx, w2_idx, w3_idx])
            target.append(float(record[3]))
        else:
            raise Exception("expected 3 token for mode=sm and 4 tokens for mode=lr, not %s tokens for mode %s" % (len(record), mode))
    return (np.asarray(idxs, dtype='int32'), np.asarray(target, dtype='int32'), token_idx)
