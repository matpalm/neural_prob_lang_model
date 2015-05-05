from __future__ import print_function
#import sys
#import numpy as np

class TokenIdx(object):  # todo: is there nothing like this in scipy/numpy ? hmm :/

    def __init__(self):
        self.token_idx = {}
        self.idx_token = {}
        self.seq = 0

    def id_exists_for(self, token):
        return token in self.token_idx

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

    def read_from_file(self, filename):
        seen_idxs = set()
        for line in open(filename, 'r').readlines():
            line = line.strip()
            cols = line.split("\t")
            if len(cols) != 2:
                raise Exception("unexpected #cols in line %s" % line)
            token, idx = cols[0], int(cols[1])
            if idx in seen_idxs:
                raise Exception("already seen idx %s" % idx)
            seen_idxs.add(idx)
            self.token_idx[token] = idx
            self.idx_token[idx] = token

        # check idxs cover full range
        if min(seen_idxs) != 0:
            raise Exception("expect idxs to start at 0")
        if max(seen_idxs) != len(seen_idxs) - 1:
            raise Exception("expected idxs to cover all numbers from 0 to #idxs. max(seed_idxs_)=%s len(seen_idxs)=%s" % (max(seen_idxs), len(seen_idxs)))

        self.seq = max(seen_idxs) + 1
