#!/usr/bin/env python

import cPickle as pickle
import gzip
import os
import sys
import time
import numpy

import theano
import theano.tensor as T
import theano.sparse as ssp

from weight_dumper import WeightDumper

from nplm import NPLM

def stats(v):
    return "mu=%.2f sd=%.2f" % (numpy.mean(v), numpy.std(v))

def run_nplm(datasets,
             n_in,
             vocab_size, projection_dim,
             n_hidden,
             input_feature_names=None,
             feature_input_indexes=None,
             learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001,
             n_epochs=100, batch_size=20, 
             seed=123, weights_file=None,
             print_internal_vars=False, 
             dump_projection=False, dump_hidden=False, dump_softmax=False,
             dump_learning_performance=False): #, do_validation_test=False):

    # alloc a "random" run id (for filenames) by taking current time.
    run_id = int(time.mktime(time.gmtime()))
    print "run_id", run_id

    # split datasets
    train_set_x, train_set_y = datasets
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    print "batch_size=%s => n_train_batches=%s" % (batch_size, n_train_batches)

    # symbolic data variables
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    rng = numpy.random.RandomState(seed)

    # construct the network
    weight_params = pickle.load(gzip.open(weights_file, 'r')) if weights_file else {}
    classifier = NPLM(rng=rng, input=x, 
                     n_in=n_in, vocab_size=vocab_size, projection_dim=projection_dim,
                     n_hidden=n_hidden, n_out=vocab_size,
                     print_internal_vars=print_internal_vars,
                     input_feature_names=input_feature_names,
                     feature_input_indexes=feature_input_indexes,
                     weight_params = weight_params)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2)
    cost = classifier.negative_log_likelihood(y) \
        + L1_reg * classifier.L1 \
        + L2_reg * classifier.L2_sqr

    # compute the gradient of cost with respect to the model
    param_gradients = T.grad(cost, classifier.params)

    # trivial update rule
    updates = []
    for param, param_gradients in zip(classifier.params, param_gradients):
        updates.append((param, param - learning_rate * param_gradients))

    # compile train_model function
    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                                  givens={x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                          y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    # some dumping objects to keep track of things
    projection_layer_dumper = WeightDumper("projection_layer_weights.%s.tsv" % run_id, 
                                           classifier.projectionLayer, 
                                           feature_names=input_feature_names)
#    hidden_layer_dumper = WeightDumper('hidden_layer_weights.tsv', classifier.hiddenLayer)
#    softmax_layer_dumper = WeightDumper('softmax_layer_weights.tsv', classifier.logRegressionLayer, node_names=input_feature_names)

    # do actual training
    epoch = 0
    last_time = time.clock()
    while (epoch < n_epochs):
        epoch = epoch + 1

        # dump all hidden weights
        if dump_hidden or epoch == n_epochs:
            projection_layer_dumper.take_snapshot(epoch)
            # hidden_layer_dumper.take_snapshot(epoch)
            # softmax_layer_dumper.take_snapshot(epoch)

        # train mini batchs
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

            # time since last 
            run_time = time.clock() - last_time
            last_time = time.clock()

            # dump stats
            if (minibatch_index % 100 == 0):
                progress_stats = "epoch %i/%i, minibatch %i/%i, runtime %.2fs     " % (epoch, n_epochs, minibatch_index + 1, n_train_batches, run_time)
                sys.stderr.write(progress_stats + "\r")

        # dump weights to a model dir
        pl_W, hl_W, hl_b, lr_W, lr_b = [p.eval() for p in classifier.params]
        weights = {'pl_W': pl_W, 'hl_W': hl_W, 'hl_b': hl_b, 'lr_W': lr_W, 'lr_b': lr_b}
        model_dir = "models/%s" % run_id
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump(weights, gzip.open("%s/nplm.%s.pkl.gz" % (model_dir, epoch), 'w'))

    # close traces
    projection_layer_dumper.close()
    # hidden_layer_dumper.close()
    # softmax_layer_dumper.close()

if __name__ == '__main__':
    # options
    import optparse
    optparser = optparse.OptionParser(prog='nplp', version='0.0.1', description='neural probabilistic language model')
    optparser.add_option('--trigrams', None, dest='trigrams', default='trigrams.pos.gz', help='trigram file')
    optparser.add_option('--n-projection-dim', None, dest='n_projection', default=3, help='num projection dimension')
    optparser.add_option('--n-hidden', None, dest='n_hidden', default=3, help='num hidden nodes')
    optparser.add_option('--l1-reg', None, dest='l1_reg', default=0, help='L1 regression weight')
    optparser.add_option('--l2-reg', None, dest='l2_reg', default=0, help='L2 regression weight')
    optparser.add_option('--batch-size', None, dest='batch_size', default=100, help='mini batch size')
    optparser.add_option('--epochs', None, dest='n_epochs', default=10000, help='epochs to run')
    optparser.add_option('--learning-rate', None, dest='learning_rate', default=0.01, help='weight update learning rate')
    optparser.add_option('--seed', None, dest='seed', default=92374652, help='rng seed')
    optparser.add_option('--weights-file', None, dest='weights_file', default=None, help='pickle model params from previous run')
    optparser.add_option('--print-internal-vars', None, action="store_true", dest='print_internal_vars', default=False, help='whether to wrap selected symbolic nodes in a theano.print')
    optparser.add_option('--dump-hidden-weights', None, action="store_true", dest='dump_hidden', default=False, help='whether to write projection/hidden/softmax layer weights out to a file')
    options, arguments = optparser.parse_args()
    print "options", options

    # load data
    from load_data import load_trigram_data
    vocab_size, input_feature_names, feature_input_indexes, datasets = load_trigram_data(options.trigrams)
    print "vocab_size", vocab_size
    print "input_feature_names", input_feature_names
    print "feature_input_indexes", feature_input_indexes
    print "datasets", datasets

    # run
    run_nplm(datasets=datasets, 
             n_in=2,  # use bigram to pick next token
             vocab_size=vocab_size, projection_dim=int(options.n_projection), 
             n_hidden=int(options.n_hidden),
             input_feature_names=input_feature_names,
             feature_input_indexes=feature_input_indexes,
             L1_reg=float(options.l1_reg), L2_reg=float(options.l2_reg),
             batch_size=int(options.batch_size), n_epochs=int(options.n_epochs),
             learning_rate=float(options.learning_rate),
             seed=int(options.seed), weights_file=options.weights_file,
             print_internal_vars=options.print_internal_vars,
             dump_hidden=options.dump_hidden,
             )
