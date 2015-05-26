#!/usr/bin/env python

# simplest-as-we-can rnn model

# - no gating within unit at all
# - no (other) protection against exploding or vanishing gradients
# - no batching, train one example at a time.
# - trivial randn weight init
# - no bias with dot products

import sys, time, optparse
import numpy as np
from util import load_training_test, perplexity_of_sequence, TokenIdx
import theano
import theano.tensor as T

optparser = optparse.OptionParser(prog='simple_rnn_model')
optparser.add_option('--adaptive-learning-rate', None, dest='adaptive_learning_rate_fn', type='string',
                     default="rmsprop", help='adaptive learning rate method')
opts, _arguments = optparser.parse_args()

training, test = load_training_test(sys.argv[1], sys.argv[2])

# data is just characters, ['A', 'B', 'A', ... ]
# but we left and right pad with <s> and </s> to include prediction of start/end of sequence
# and convert to idxs
vocab = TokenIdx()
def convert_to_ids(vocab, data):
    return [vocab.ids_for(["<s>"] + sequence + ["</s>"]) for sequence in data]
training = convert_to_ids(vocab, training)
test = convert_to_ids(vocab, test)

# matrix sizing
n_in = vocab.num_entries()
n_hidden = 10

# t_x input and t_y output sequence
t_x = T.ivector('x')  # eg s A B A D   for sequence A B A D
t_y = T.ivector('y')  # eg A B A D /s  for sequence A B A D

# initial hidden state for each sequence. always zero so just stored as shared.
t_h0 = theano.shared(np.zeros(n_hidden, dtype='float32'), name='h0', borrow=True)

# learnt weights. init for now with simple randn.
# TODO: orthogonalize? or just identity (le paper)
t_Wx = theano.shared(np.asarray(np.random.randn(n_hidden, n_in), dtype='float32'),
                     name='Wx', borrow=True)
t_Wrec = theano.shared(np.asarray(np.random.randn(n_hidden, n_hidden), dtype='float32'),
                     name='Wrec', borrow=True)
t_Wy = theano.shared(np.asarray(np.random.randn(n_in, n_hidden), dtype='float32'),
                     name='Wy', borrow=True)

def recurrent_step(x_t, h_t0):
    # calc new hidden state; elementwise add of embedded input & 
    # recurrent weights dot _last_ hiddenstate
    h_t = T.nnet.sigmoid(t_Wx[:, x_t] + T.dot(t_Wrec, h_t0))
    # note: h_t1 = T.tanh and no clipping explodes _very_ quickly! :/

    # calc output; softmax over output weights dot hidden state
    y_t = T.flatten(T.nnet.softmax(T.dot(t_Wy, h_t)), 1)

    # return what we want to have per output step
    return [h_t, y_t]

# define the recurrence
[h_ts, t_y_softmax], _ = theano.scan(fn=recurrent_step,
                                     sequences=[t_x],
                                     outputs_info=[t_h0, None])

# loss is just cross entropy of the softmax output compared to the target
cross_entropy = T.mean(T.nnet.categorical_crossentropy(t_y_softmax, t_y))

# gradient update; either vanilla or rmsprop
learning_rate = 0.05

def vanilla(params, gradients):
    return [(param, param - learning_rate * gradient) for param, gradient in zip(params, gradients)]

def rmsprop(params, gradients):
    updates = []
    for param_t0, gradient in zip(params, gradients):
        # rmsprop see slide 29 of http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        # first the mean_sqr exponential moving average
        # TODO: hmmm. current checkpointer doesn't support this at all :(
        mean_sqr_t0 = theano.shared(np.zeros(param_t0.get_value().shape, dtype=param_t0.get_value().dtype))  # zeros in same shape are param
        mean_sqr_t1 = 0.9 * mean_sqr_t0 + 0.1 * (gradient**2)
        updates.append((mean_sqr_t0, mean_sqr_t1))
        # update param surpressing gradient by this average
        param_t1 = param_t0 - learning_rate * (gradient / T.sqrt(mean_sqr_t1 + 1e-10))
        updates.append((param_t0, param_t1))
    return updates

# lookup update fn from opts
update_fn = globals().get(opts.adaptive_learning_rate_fn)
if update_fn == None:
    raise Exception("no update_fn " + opts.adaptive_learning_rate_fn)

params = [t_Wx, t_Wrec, t_Wy]
gradients = T.grad(cost=cross_entropy, wrt=params)
updates = update_fn(params, gradients)


compile_start_time = time.time()

# compile function for training; ie with backprop updates
train_fn = theano.function(inputs=[t_x, t_y],
                           outputs=[cross_entropy],
                           updates=updates)

# compile function to emit predictions
predict_fn = theano.function(inputs=[t_x],
                             outputs=t_y_softmax)  # full distribution

print "compilation took %0.3f s" % (time.time()-compile_start_time)

for epoch in range(5):
    start_time = time.time()

    # train all examples. no batching yet!! o_O
    for n, training_eg in enumerate(training):
        x, y = training_eg[:-1], training_eg[1:]
        cost, = train_fn(x, y)
        #print "COST\t%s" % cost

    # eval test set (again no batching)
    perplexities = []
    for test_eg in test:
        probabilities = []
        x, y = test_eg[:-1], test_eg[1:]
        y_softmaxs = predict_fn(x)
        for y_true_i, y_softmax in zip(y, y_softmaxs):
            y_true_confidence = y_softmax[y_true_i]
            probabilities.append(y_true_confidence)
        perplexities.append(perplexity_of_sequence(probabilities))

    print "epoch", epoch,
    print "min, mean, max perplexity", " ".join(["%.3f"%s for s in [min(perplexities), np.mean(perplexities), max(perplexities)]]),
    print "took %.3f sec" % (time.time()-start_time)
