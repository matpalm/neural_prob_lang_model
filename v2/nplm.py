#!/usr/bin/env python
from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np
import random, math, time, optparse, sys

from load_data import load_trigram_data
from matrix_dumper import MatrixDumper

optparser = optparse.OptionParser(prog='nplm', version='0.0.1', description='simple neural probabilistic language model')
optparser.add_option('--mode', None, dest='mode', type='string', default="sm", help='top layer mode; sm=softmax, lr=logistic regression')
optparser.add_option('--trigrams', None, dest='trigrams_file', type='string', default="trigrams.txt", help='trigrams input file')
optparser.add_option('--batch-size', None, dest='batch_size', type='int', default=100, help='training batch size')
optparser.add_option('--embedding-dim', None, dest='embedding_dim', type='int', default=3, help='embedding dimensionality')
optparser.add_option('--n-hidden', None, dest='n_hidden', type='int', default=3, help='num hidden nodes')
optparser.add_option('--seed', None, dest='seed', type='int', default=None, help='rng seed')
optparser.add_option('--epochs', None, dest='epochs', type='int', default=20000, help='epochs to run')
optparser.add_option('--lambda1', None, dest='lambda1', type='float', default=0.0, help='l1 regularisation weight')
optparser.add_option('--lambda2', None, dest='lambda2', type='float', default=0.0, help='l2 regularisation weight')
optparser.add_option('--adaptive-learning-rate', None, dest='adaptive_learning_rate_fn', type='string', default="rmsprop", help='adaptive learning rate method')
optparser.add_option('--learning-rate', None, dest='learning_rate', type='float', default=0.001, help='base learning rate')
optparser.add_option('--output-file-prefix', None, dest='output_file_prefix', default="", help='prefix for all output files')
optparser.add_option('--cost-progress-freq', None, dest='cost_progress_freq', default=10, type=int, help='how frequently to write cost output (in batchs) 0==never')
optparser.add_option('--dump-matrices-freq', None, dest='dump_matrices_freq', default=None, type=int, help='how frequently to dump matrices (in batchs) 0==never')

opts, arguments = optparser.parse_args()
print("options", opts, file=sys.stderr)
if opts.seed is not None:
    np.random.seed(int(opts.seed))
if opts.mode not in ['sm', 'lr']:
    raise Exception("mode must be one of 'sm' or 'lr' not '%s'" % opts.mode)

# slurp in training data, converting from "C A B" to idx "0 1 2" and storing in idxs
# idxs => (w1, w2, w3) for lr; (w1, w2) for sm
# label y => 1.0 or 0.0 for lr; w3 for sm
idxs, y, token_idx = load_trigram_data(opts.trigrams_file, opts.mode)
token_idx.write_to_file("vocab.tsv")
VOCAB_SIZE = token_idx.seq

# decide batching sizes
BATCH_SIZE = opts.batch_size
NUM_BATCHES = int(math.ceil(float(len(idxs)) / BATCH_SIZE))
print("#egs", len(idxs), "batch_size", BATCH_SIZE, "=> num_batches", NUM_BATCHES, file=sys.stderr)
EPOCHS = opts.epochs
LAMBDA1, LAMBDA2 = opts.lambda1, opts.lambda2

# embeddings matrix
E = np.asarray(np.random.randn(VOCAB_SIZE, opts.embedding_dim), dtype='float32')   # orthogonise with U (V?) from np.linalg.svd
NUM_EMBEDDING_NODES = 3 if opts.mode=='lr' else 2
# hidden layer weights and bias
hW = np.asarray(np.random.randn(NUM_EMBEDDING_NODES * opts.embedding_dim, opts.n_hidden), dtype='float32')
hB = np.zeros(opts.n_hidden, dtype='float32')
# output weights and bias
NUM_OUTPUT_NODES = 1 if opts.mode=='lr' else VOCAB_SIZE
oW = np.asarray(np.random.randn(opts.n_hidden, NUM_OUTPUT_NODES), dtype='float32')
oB = np.zeros(NUM_OUTPUT_NODES, dtype='float32')

# build up network
t_idxs = T.imatrix(name='idxs')  # input egs
t_y = T.ivector(name='y')        # input labels
# embedding layer
t_E = theano.shared(E, name='E', borrow=True)
t_embedding_output = t_E[t_idxs].reshape((t_idxs.shape[0], -1))
# hidden layer
t_hW = theano.shared(hW, name='hW', borrow=True)
t_hB = theano.shared(hB, name='hB', borrow=True)
t_hidden_layer_output = T.tanh(T.add(T.dot(t_embedding_output, t_hW), t_hB))
# output layer; sigmoid for logisitic regression, softmax for, err, softmax
t_oW = theano.shared(oW, name='oW', borrow=True)
t_oB = theano.shared(oB, name='oB', borrow=True)
output_activation_function = T.nnet.sigmoid if opts.mode=='lr' else T.nnet.softmax
p_y_given_x = output_activation_function(T.dot(t_hidden_layer_output, t_oW) + t_oB)
#softmax_output = T.argmax(p_y_given_x, axis=1)

# params to regularise; not embeddings, nor biases.
regularised_params = [t_hW, t_oW]
L1 = T.sum([T.sum(abs(p)) for p in regularised_params])
L2 = T.sum([T.sum(p*p) for p in regularised_params])

# cost 
if opts.mode == 'lr':
    per_element_cost = T.nnet.binary_crossentropy(p_y_given_x.T, t_y)
else:  # 'sm'
    #per_element_cost = -T.log(p_y_given_x)[T.arange(t_y.shape[0]), t_y]  # negative_log_likelihood
    per_element_cost = T.nnet.categorical_crossentropy(p_y_given_x, t_y)
cost = T.mean(per_element_cost) + (LAMBDA1 * L1) + (LAMBDA2 * L2)

learning_rate = theano.shared(np.float32(opts.learning_rate))

def vanilla(params, gradients):
    return [(param, param - learning_rate * gradient) for param, gradient in zip(params, gradients)]

def rmsprop(params, gradients):
    updates = []
    for param_t0, gradient in zip(params, gradients):
        # rmsprop see slide 29 of http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        # first the mean_sqr exponential moving average
        mean_sqr_t0 = theano.shared(np.zeros(param_t0.get_value().shape, dtype=param_t0.get_value().dtype))  # zeros in same shape are param
        mean_sqr_t1 = 0.9 * mean_sqr_t0 + 0.1 * (gradient**2)
        updates.append((mean_sqr_t0, mean_sqr_t1))
        # update param surpressing gradient by this average
        param_t1 = param_t0 - learning_rate * (gradient / T.sqrt(mean_sqr_t1 + 1e-10))
        updates.append((param_t0, param_t1))
    return updates

update_fn = globals().get(opts.adaptive_learning_rate_fn)
if update_fn == None:
    raise Exception("no update_fn " + opts.adaptive_learning_rate_fn)

params = [t_E, t_hW, t_hB, t_oW, t_oB]
gradients = T.grad(cost=cost, wrt=params)
updates = update_fn(params, gradients)

train_model = theano.function(inputs=[t_idxs, t_y], outputs=[], updates=updates)
check_cost = theano.function(inputs=[t_idxs, t_y], outputs=[cost])
model_output = theano.function(inputs=[t_idxs], outputs=[p_y_given_x])

#theano.printing.pydotprint(train_model, outfile="model.png")
#theano.printing.pydotprint(gradients, outfile="gradients.png")

# debugging wrappers to dump weights over time
embeddings_dumper = MatrixDumper(opts.output_file_prefix + "_embeddings.tsv", t_E, token_idx)
hidden_weights = None  #MatrixDumper("hidden_weights.tsv", t_hW)

def print_likelihood_of(i, ws):
    idxs = [[token_idx.id_for(w) for w in ws]]  # batch size of 1
    if opts.mode=='lr':
        print("%s\t%s %s %s\t%s" % (i, ws[0], ws[1], ws[2], model_output(idxs)[0][0][0]))  # no, i'm not joking :/
    else:  # 'sm'
        distr = model_output(idxs)[0][0]
        for idx, prob in enumerate(distr):
            w3 = token_idx.token_for(idx)
            print("%s\t%s %s %s\t%s" % (i, ws[0], ws[1], w3, prob))

cost_progress_file = open(opts.output_file_prefix + "_cost.tsv", 'w')  # move into cost_dumper
print("e\tb\tcost\ttime", file=cost_progress_file)

last_time = time.time()
total_batches_run = 0

for e in range(EPOCHS):
    for b in range(NUM_BATCHES):

        # train one batch
        batch_idxs = idxs[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
        batch_y = y[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
        train_model(batch_idxs, batch_y)

        # print some cost stats
        if total_batches_run % opts.cost_progress_freq == 0:
            time_since_last = time.time() - last_time
            cost = check_cost(batch_idxs, batch_y)
            sys.stderr.write("e:%s/%s b:%s/%s cost:%s last_time:%s                  \r" % (e, EPOCHS, b, NUM_BATCHES, cost[0], time_since_last))
            print("\t".join(map(str, [e, b, cost[0], time_since_last])), file=cost_progress_file)
            cost_progress_file.flush()
            last_time = time.time()

        # dump embeddings and weights to file
        if total_batches_run % opts.dump_matrices_freq == 0:
            for md in [embeddings_dumper, hidden_weights]:
                if md is not None:
                    md.dump(e, b)

        total_batches_run += 1

# dump a couple of known cases (see blog post for freq analysis)
#for w1w2 in ['FA', 'CB', 'CC']:
#    if opts.mode=='lr':
#        for w3 in "ABCDEF":
#            print_likelihood_of(i, w1w2 + w3)
#    else: # sm
#        print_likelihood_of(i, w1w2)
