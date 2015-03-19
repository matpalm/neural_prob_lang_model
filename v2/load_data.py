import numpy as np

class TokenIdx(object):  # todo: is there nothing like this in scipy/numpy ? hmm :/

    def __init__(self):
        self.token_idx = {}
        self.idx_token = {}
        self.tokens = set()
        self.seq = 0

    def id_for(self, token):
        if token in self.tokens:
            return self.token_idx[token]
        else:
            self.token_idx[token] = self.seq
            self.idx_token[self.seq] = token
            self.tokens.add(token)
            self.seq += 1
            return self.seq - 1

    def token_for(self, idx):
        return self.idx_token[idx]

    def labels(self):
        return [self.idx_token[i] for i in range(self.seq)]

def load_trigram_data(filename):
    token_idx = TokenIdx()
    idxs = [] 
    target = []
    for line in open(filename, 'r'):
        record = line.strip().split()
        w1_idx, w2_idx, w3_idx = [token_idx.id_for(w) for w in record[:3]]
        if len(record) == 3:
            # just trigram
            idxs.append([w1_idx, w2_idx])
            target.append(w3_idx)
        elif len(record) == 4:
            # trigram with label
            idxs.append([w1_idx, w2_idx, w3_idx])
            target.append(float(record[3]))
    return (np.asarray(idxs, dtype='int32'), np.asarray(target, dtype='int32'), token_idx)
