import cPickle
import gzip
import os
import sys
import time
import numpy
import theano
import theano.tensor as T
import random
from sklearn.preprocessing import *

# taken from http://deeplearning.net/tutorial/code/logistic_sgd.py
def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),
                             borrow=borrow)
    # cast to int32 to be able to use float32 (required for gpu) as array indexes
    return shared_x, T.cast(shared_y, 'int32')


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

def load_trigram_data(filename):
    # slurp trigrams
    token_idx = TokenIdx()
    data = []   # raw token indexes (not 1hots)
    target = [] 
    for line in gzip.open(filename, 'r'):
        trigram = line.strip().split()
        w1_idx, w2_idx, w3_idx = [token_idx.id_for(w) for w in trigram] 
        # print "trigram %s => data=[%s, %s] target=%s" % (trigram, w1_idx, w2_idx, w3_idx)
        data.append([w1_idx, w2_idx])
        target.append(w3_idx)

    dataset = shared_dataset((data, target))
    print "data", len(data)

    vocab_size = token_idx.seq
    return (vocab_size, token_idx.idx_token, token_idx.token_idx, dataset)
