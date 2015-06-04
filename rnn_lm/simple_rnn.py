#!/usr/bin/env python
import numpy as np
import util
import theano
import theano.tensor as T

class SimpleRnn(object):
    def __init__(self, n_in, n_hidden):
        self.t_Wx = util.sharedMatrix(n_hidden, n_in, 'Wx')
        self.t_Wrec = util.sharedMatrix(n_hidden, n_hidden, 'Wrec')
        self.t_Wy = util.sharedMatrix(n_in, n_hidden, 'Wy')

    def params(self):
        return [self.t_Wx, self.t_Wrec, self.t_Wy]

    def recurrent_step(self, x_t, h_t0):
        # calc new hidden state; elementwise add of embedded input & 
        # recurrent weights dot _last_ hiddenstate
        h_t = T.nnet.sigmoid(self.t_Wx[:, x_t] + T.dot(self.t_Wrec, h_t0))

        # calc output; softmax over output weights dot hidden state
        y_t = T.flatten(T.nnet.softmax(T.dot(self.t_Wy, h_t)), 1)

        # return what we want to have per output step
        return [h_t, y_t]

    def t_y_softmax(self, t_x, t_h0):
        [_h_ts, t_y_softmax], _ = theano.scan(fn=self.recurrent_step,
                                             sequences=[t_x],
                                             outputs_info=[t_h0, None])
        return t_y_softmax
