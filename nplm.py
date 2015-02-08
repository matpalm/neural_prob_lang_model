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

# basically a copy of http://deeplearning.net/tutorial/code/logistic_sgd.py
class LogisticRegression(object):

    def __init__(self, input, n_in, n_out, 
                 W=None, b=None,
                 print_internal_vars=False):
        if W == None:
            W = numpy.zeros((n_in, n_out), dtype=theano.config.floatX)
        self.W = theano.shared(value=W, name='W', borrow=True)

        if b == None:
            b = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b, name='b', borrow=True)

        linear_output = T.dot(input, self.W) + self.b
        if print_internal_vars:
            linear_output = theano.printing.Print('output pre softmax p_y_given_x')(linear_output)
        p_y_given_x = T.nnet.softmax(linear_output)
        if print_internal_vars:
            p_y_given_x = theano.printing.Print('output softmaxed p_y_given_x')(p_y_given_x)
        self.p_y_given_x = p_y_given_x

        y_pred = T.argmax(self.p_y_given_x, axis=1)
        if print_internal_vars:
            y_pred = theano.printing.Print('output argmax y_pred')(y_pred)
        self.y_pred = y_pred

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            y_pred = self.y_pred
            if print_internal_vars:
                y_pred= theano.printing.Print('errors y_pred')(y_pred)
            return T.mean(T.neq(y_pred, y))
        else:
            raise NotImplementedError()

# basically a copy of Hidden layer from http://deeplearning.net/tutorial/code/mlp.py
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, print_internal_vars=False):
        self.input = input

        if W == None: 
            # See Glorot & Bengio, 2010, "Understanding the difficulty of training deep feedforward neural networks"
            W_weight_range = numpy.sqrt(6. / (n_in + n_out))
            W = numpy.asarray(rng.uniform(
                    low=-W_weight_range,
                    high=W_weight_range,
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W *= 4
        self.W = theano.shared(value=W, name='W', borrow=True)

        if b == None:
            b = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b, name='b', borrow=True)

        linear_output = T.dot(input, self.W) + self.b
        output = linear_output if activation is None else activation(linear_output)
        if print_internal_vars:
            output = theano.printing.Print('output of hidden layer')(output)
        self.output = output

        self.params = [self.W, self.b]

# a token embedding layer
# see Bengio et al, 2003, "A Neural Probabilistic Language Model"
class ProjectionLayer(object):
    def __init__(self, rng, input, vocab_size, projection_dim, W=None,
                 print_internal_vars=False):
        """
        :type vocab_size: int
        :param vocab_size: |V|.  W.shape = (|V|, d)

        :type projection_dim: int
        :param projection_dim: projection dimension, d.  W.shape = (|V|, d)
        """        
        if W == None:
            W = numpy.asarray(rng.uniform(low=-1, high=1, size=(vocab_size, projection_dim)),
                                          dtype=theano.config.floatX)
        self.W = theano.shared(value=W, name='W', borrow=True)

        # input is a pair of indexes; w1 and w2
        if print_internal_vars:
            input = theano.printing.Print('input to projection layer')(input)
        self.input = input 

        # # @aboSamoor's sparse matrix dot product
        # # https://groups.google.com/forum/#!searchin/theano-users/one$20hot$20vector/theano-users/lobCNFMlMeA/hUoNUb270N4J
        # data_flat = input.flatten()
        # data_ones = T.cast(T.ones_like(data_flat), config.floatX)
        # shape1 = T.as_tensor_variable([data_ones.shape[0], self.W.shape[0]])
        # indptr = T.arange(data_ones.shape[0]+1)
        # m1 = ssp.CSR(data_ones, data_flat, indptr, shape1)
        # m2 = ssp.dot(m1, self.W)
        # self.output = m2.flatten().reshape((data.shape[0], self.W.shape[1] * data.shape[1]))

        # output is concatenation of W rows for w1, w2 indexes
        indexed_rows = self.W[T.cast(input, 'int32')] 
        concatenated_rows = indexed_rows.flatten()
        num_examples = input.shape[0]
        width = concatenated_rows.size // num_examples
        output = concatenated_rows.reshape((num_examples, width))
        if print_internal_vars:
            output = theano.printing.Print('output of projection layer')(output)
        self.output = output

        self.params = [self.W]


class NPLM(object):

    def __init__(self, rng, input, n_in, vocab_size, projection_dim, n_hidden, n_out, 
                 print_internal_vars, input_feature_names, feature_input_indexes,
                 weight_params = None):

        # keep track of feature name -> index mappings
        self.input_feature_names = input_feature_names
        self.feature_input_indexes = feature_input_indexes

        # token embedding layer
        self.projectionLayer = ProjectionLayer(rng=rng, input=input,
                                               vocab_size=vocab_size, projection_dim=projection_dim,
                                               print_internal_vars=print_internal_vars,
                                               W=weight_params.get('pl_W'))

        # single hidden layer
        self.hiddenLayer = HiddenLayer(rng=rng, input=self.projectionLayer.output,
                                       n_in=projection_dim * n_in,  # projection layer concats word embeddings
                                       n_out=n_hidden,
                                       activation=T.tanh,
                                       print_internal_vars=print_internal_vars,
                                       W=weight_params.get('hl_W'), b=weight_params.get('hl_b'))

        # final softmax logistic regression
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out, 
            print_internal_vars=print_internal_vars,
            W=weight_params.get('lr_W'), b=weight_params.get('lr_b'))

        # L1 / L2 norm regularizations
        # note: no regularization of embedding space.
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood and errors of entire model are those of the logistic layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors

        self.params = self.projectionLayer.params + \
                      self.hiddenLayer.params + \
                      self.logRegressionLayer.params

        # a function for getting the prediction distribution for a bigram 
        self.predict_f = theano.function(inputs=[input],
                                         outputs=self.logRegressionLayer.p_y_given_x)  

    def predict(self, w1, w2):
        # map token -> idxs
        i1, i2 = [self.feature_input_indexes[x] for x in [w1, w2]]
        test_input = numpy.asarray([[i1, i2]], dtype=theano.config.floatX)
        # map call to network
        per_index_predictions = self.predict_f(test_input)[0]
        # map _back_ to features by name
        per_feature_predictions = [(self.input_feature_names[idx], p) for idx, p in enumerate(per_index_predictions)]
        # return results, sorted by prob
        return sorted(per_feature_predictions, key=lambda (feat, prob): -prob)

