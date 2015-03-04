#!/usr/bin/env python
import theano
import theano.tensor as T
import numpy as np
import random, math, time, optparse, sys

from load_data import load_trigram_data
from matrix_dumper import MatrixDumper

optparser = optparse.OptionParser(prog='nplp', version='0.0.1', description='simple neural probabilistic language model')
optparser.add_option('--trigrams', None, dest='trigrams_file', type='string', default="trigrams.txt", help='trigrams input file')
optparser.add_option('--batch-size', None, dest='batch_size', type='int', default=100, help='training batch size')
optparser.add_option('--embedding-dim', None, dest='embedding_dim', type='int', default=3, help='embedding dimensionality')
optparser.add_option('--n-hidden', None, dest='n_hidden', type='int', default=3, help='num hidden nodes')
optparser.add_option('--seed', None, dest='seed', type='int', default=None, help='rng seed')
optparser.add_option('--epochs', None, dest='epochs', type='int', default=20000, help='epochs to run')
opts, arguments = optparser.parse_args()
print "options", opts
if opts.seed is not None:
    np.random.seed(int(opts.seed))

# slurp in training data, converting from "A B C" to idx "0 1 2"
idxs, y, token_idx = load_trigram_data(opts.trigrams_file)
VOCAB_SIZE = token_idx.seq

# decide batching sizes
BATCH_SIZE = opts.batch_size
NUM_BATCHES = int(math.ceil(float(len(idxs)) / BATCH_SIZE))
print "#egs", len(idxs), "batch_size", BATCH_SIZE, "=> num_batches", NUM_BATCHES
EPOCHS = opts.epochs

# embeddings matrix
print >>sys.stderr, "generating params"
E = np.asarray(np.random.randn(VOCAB_SIZE, opts.embedding_dim), dtype='float32')
#print "E", E
# hidden layer weights and bias
hW = np.asarray(np.random.randn(2 * opts.embedding_dim, opts.n_hidden), dtype='float32')
hB = np.zeros(opts.n_hidden, dtype='float32')
#print "hW", hW
#print "hB", hB
# softmax weights and bias
smW = np.asarray(np.random.randn(opts.n_hidden, VOCAB_SIZE), dtype='float32')
smB = np.zeros(VOCAB_SIZE, dtype='float32')
#print "smW", smW
#print "smB", smB

# build up network
print >>sys.stderr, "wrapping parms as shared T data"
t_idxs = T.imatrix(name='idxs')  # input eg(s)
t_y = T.ivector(name='y')        # input label(s)
# embedding layer
t_E = theano.shared(E, name='E', borrow=True)
t_embedding_output = t_E[t_idxs].reshape((t_idxs.shape[0], -1))
# hidden layer
t_hW = theano.shared(hW, name='hW', borrow=True)
t_hB = theano.shared(hB, name='hB', borrow=True)
t_hidden_layer_output = T.tanh(T.add(T.dot(t_embedding_output, t_hW), t_hB))
# softmax
t_smW = theano.shared(smW, name='smW', borrow=True)
t_smB = theano.shared(smB, name='smW', borrow=True)
p_y_given_x = T.nnet.softmax(T.dot(t_hidden_layer_output, t_smW) + t_smB)
softmax_output = T.argmax(p_y_given_x, axis=1)

# cost => gradient updates => compiled training method
print >>sys.stderr, "compiling training update"
negative_log_likelihood = -T.mean(T.log(p_y_given_x)[T.arange(t_y.shape[0]), t_y])
params = [t_E, t_hW, t_hB, t_smW, t_smB]
gradients = T.grad(cost=negative_log_likelihood, wrt=params)
updates = []
for param, gradient in zip(params, gradients):
    updates.append((param, param - 0.01 * gradient))
train_model = theano.function(inputs=[t_idxs, t_y], outputs=[negative_log_likelihood], updates=updates)
softmax_distribution_model = theano.function(inputs=[t_idxs], outputs=[p_y_given_x])
#print theano.printing.debugprint(train_model)

# debugging wrappers to dump weights over time
embeddings_dumper = MatrixDumper("embeddings.tsv", t_E, token_idx)
hidden_weights = MatrixDumper("hidden_weights.tsv", t_hW)

def distribution_for(w1, w2):
    idxs = [[token_idx.id_for(w) for w in [w1, w2]]]
    distr, = softmax_distribution_model(idxs)
    distr = list(reversed(sorted(zip(distr[0], token_idx.labels()))))[:4]
    return "P(W3|(w1=%s,w2=%s)=%s" % (w1, w2, distr)

print >>sys.stderr, "training"
start = time.time()
i = 0
for e in range(EPOCHS):
    for b in range(NUM_BATCHES):
        batch_idxs = idxs[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
        batch_y = y[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
        negative_log_likelihood = train_model(batch_idxs, batch_y)

        if i % 5000 == 0:
            [d.dump(e, b, i) for d in [embeddings_dumper, hidden_weights]]
            print "e", e, "b", b, "i", i
            print "negative_log_likelihood", negative_log_likelihood
            print distribution_for('A1', 'B1')
            print distribution_for('B1', 'C1')
            print distribution_for('C1', 'D1')
            print distribution_for('D1', 'A1')
            print distribution_for('A1', 'C1')
            print distribution_for('A1', 'D1')
            print distribution_for('A1', 'A2')
        i += 1

print "runtime", time.time() - start
embeddings_dumper.dump(EPOCHS, 0, i)



