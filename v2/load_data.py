import numpy as np

class TokenIdx(object):  # todo: is there nothing like this in scipy/numpy ? hmm :/

    def __init__(self):
        self.token_idx = {}
        self.idx_token = {}  # debug only
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

    def labels(self):
        return [self.idx_token[i] for i in range(self.seq)]

def load_trigram_data(filename):
    token_idx = TokenIdx()
    idxs = [] 
    target = [] 
    for line in open(filename, 'r'):
        trigram = line.strip().split()
        w1_idx, w2_idx, w3_idx = [token_idx.id_for(w) for w in trigram] 
        # print "trigram %s => data=[%s, %s] target=%s" % (trigram, w1_idx, w2_idx, w3_idx)
        idxs.append([w1_idx, w2_idx])
        target.append(w3_idx)
    return (np.asarray(idxs, dtype='int32'), np.asarray(target, dtype='int32'), token_idx)
