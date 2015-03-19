#!/usr/bin/env python
import theano
import theano.tensor as T
import numpy as np
import random, math, time, optparse, sys

from load_data import load_trigram_data
from matrix_dumper import MatrixDumper

optparser = optparse.OptionParser(prog='nplp', version='0.0.1', description='simple neural probabilistic language model')
optparser.add_option('--trigrams', None, dest='trigrams_file', type='string', default="trigrams.with_negs.txt", help='trigrams input file')
optparser.add_option('--batch-size', None, dest='batch_size', type='int', default=100, help='training batch size')
optparser.add_option('--embedding-dim', None, dest='embedding_dim', type='int', default=3, help='embedding dimensionality')
optparser.add_option('--n-hidden', None, dest='n_hidden', type='int', default=3, help='num hidden nodes')
optparser.add_option('--seed', None, dest='seed', type='int', default=None, help='rng seed')
optparser.add_option('--epochs', None, dest='epochs', type='int', default=20000, help='epochs to run')
opts, arguments = optparser.parse_args()
print >>sys.stderr, "options", opts
if opts.seed is not None:
    np.random.seed(int(opts.seed))

# slurp in training data, converting from "C A B" to idx "0 1 2" and storing in idxs
# idxs => (w1, w2, w3)
# label y => 1.0 or 0.0
idxs, y, token_idx = load_trigram_data(opts.trigrams_file)
VOCAB_SIZE = token_idx.seq

# for debugging build set of known distinct (wi) & (w1, w2) in training data
distinct_w = set()
distinct_w1_w2_w3 = set()
for w_123 in idxs:
    t_123 = [token_idx.idx_token[w] for w in w_123]
    distinct_w1_w2_w3.add(tuple(t_123))
    distinct_w.update(t_123)

# decide batching sizes
BATCH_SIZE = opts.batch_size
NUM_BATCHES = int(math.ceil(float(len(idxs)) / BATCH_SIZE))
print >>sys.stderr, "#egs", len(idxs), "batch_size", BATCH_SIZE, "=> num_batches", NUM_BATCHES
EPOCHS = opts.epochs

# embeddings matrix
E = np.asarray(np.random.randn(VOCAB_SIZE, opts.embedding_dim), dtype='float32')
# hidden layer weights and bias
hW = np.asarray(np.random.randn(3 * opts.embedding_dim, opts.n_hidden), dtype='float32')
hB = np.zeros(opts.n_hidden, dtype='float32')
# final output logisitic regression weights and bias
lrW = np.asarray(np.random.randn(opts.n_hidden, 1), dtype='float32')
lrB = np.zeros(1, dtype='float32')

# build up network
t_idxs = T.imatrix(name='idxs')  # input eg(s); w1 w2 w3
t_y = T.ivector(name='y')        # input label(s); 1.0 or 0.0
# embedding layer
t_E = theano.shared(E, name='E', borrow=True)
t_embedding_output = t_E[t_idxs].reshape((t_idxs.shape[0], -1))
# hidden layer
t_hW = theano.shared(hW, name='hW', borrow=True)
t_hB = theano.shared(hB, name='hB', borrow=True)
t_hidden_layer_output = T.tanh(T.add(T.dot(t_embedding_output, t_hW), t_hB))
# single output logistic regression
t_lrW = theano.shared(lrW, name='lrW', borrow=True)
t_lrB = theano.shared(lrB, name='lrB', borrow=True)
p_y_given_x = T.nnet.sigmoid(T.dot(t_hidden_layer_output, t_lrW) + t_lrB).T

# cost => gradient updates => compiled training method
cost = T.mean(T.nnet.binary_crossentropy(p_y_given_x, t_y))

# trying to explicitly lay this out results in NaNs? but formula the same? (?)
# there seems to be some numerical stability trick/optimisation that's not coming into play...
#cost = T.mean((p_y_given_x * T.log(t_y)) + (1-p_y_given_x * T.log(1-t_y)))

params = [t_E, t_hW, t_hB, t_lrW, t_lrB]
gradients = T.grad(cost=cost, wrt=params)

# TODO: seriously need RMSprop, adagrad, whatever
updates = []
for param, gradient in zip(params, gradients):
    updates.append((param, param - 0.001 * gradient))

train_model = theano.function(inputs=[t_idxs, t_y], outputs=[], updates=updates)
check_cost = theano.function(inputs=[t_idxs, t_y], outputs=[cost])
p_y = theano.function(inputs=[t_idxs], outputs=[p_y_given_x])

#theano.printing.pydotprint(train_model, outfile="model.png")
#theano.printing.pydotprint(gradients, outfile="gradients.png")

# debugging wrappers to dump weights over time
embeddings_dumper = MatrixDumper("embeddings.tsv", t_E, token_idx)
hidden_weights = MatrixDumper("hidden_weights.tsv", t_hW)

def print_likelihood_of(i, ws):
    idxs = [[token_idx.id_for(w) for w in ws]]  # batch size of 1
    print "\t".join(map(str, [i, " ".join(ws), p_y(idxs)[0][0][0]]))  # no, i'm not joking :/

last_batch_start = time.time()
i = 0  # a counter primarily for dumping weights to file

for e in range(EPOCHS):
    # TODO shuffle egs

    # run all batches
    for b in range(NUM_BATCHES):
        batch_idxs = idxs[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
        batch_y = y[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
        train_model(batch_idxs, batch_y)

    if e % 100 == 0:
        # print some stats
        i += 1
        cost = check_cost(batch_idxs, batch_y)        
        print >>sys.stderr, "e", e, "i", i, "cost", cost[0], "last_batch_time", time.time() - last_batch_start
        last_batch_start = time.time()

        # dump embeddings and weights to file
        for md in [embeddings_dumper, hidden_weights]:
            if md is not None:
                md.dump(e, b, i)    

# dump a couple of known cases (see blog post for freq analysis)
for w1w2 in ['FA', 'CB', 'CC']:
    for w3 in "ABCDEF":
        print_likelihood_of(i, w1w2 + w3)
