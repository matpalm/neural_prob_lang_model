from __future__ import print_function
import sys
import numpy as np

class TokenIdx(object):  # todo: is there nothing like this in scipy/numpy ? hmm :/

    def __init__(self):
        self.token_idx = {}
        self.idx_token = {}
        self.seq = 0

    def id_for(self, token):
        if token in self.token_idx:
            return self.token_idx[token]
        else:
            self.token_idx[token] = self.seq
            self.idx_token[self.seq] = token
            self.seq += 1
            return self.seq - 1

    def token_for(self, idx):
        return self.idx_token[idx]

    def labels(self):
        return [self.idx_token[i] for i in range(self.seq)]

    def write_to_file(self, filename):
        with open(filename, 'w') as f:
            for token, idx in self.token_idx.iteritems():
                print("%s\t%s" % (token, idx), file=f)


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
