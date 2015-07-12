#!/usr/bin/env python
import numpy as np
import util
import theano
import theano.tensor as T

class GruRnn(object):
    def __init__(self, n_in, n_hidden):
        self.t_Wx = util.sharedMatrix(n_hidden, n_in, 'Wx')
        self.t_Wz = util.sharedMatrix(n_hidden, n_in, 'Wz')
        self.t_Wr = util.sharedMatrix(n_hidden, n_in, 'Wr')
        self.t_Ux = util.sharedMatrix(n_hidden, n_hidden, 'Ux')
        self.t_Uz = util.sharedMatrix(n_hidden, n_hidden, 'Uz')
        self.t_Ur = util.sharedMatrix(n_hidden, n_hidden, 'Ur')
        self.t_Wy = util.sharedMatrix(n_in, n_hidden, 'Wy')

    def params(self):
        return [self.t_Wx, self.t_Wz, self.t_Wr, self.t_Ux, self.t_Uz, self.t_Ur, self.t_Wy]

    def recurrent_step(self, x_t, h_t0):
        # calc reset gate activation
        r = T.nnet.sigmoid(self.t_Wr[:, x_t] + T.dot(self.t_Ur, h_t0))
        # calc candidate next hidden state (with reset)
        h_t_candidate = T.tanh(self.t_Wx[:, x_t] + T.dot(self.t_Ux, r * h_t0))

        # calc update gate activation 
        z = T.nnet.sigmoid(self.t_Wz[:, x_t] + T.dot(self.t_Uz, h_t0))
        # calc hidden state as affine combo of last state and candidate next state
        h_t = (1 - z) * h_t0 + z * h_t_candidate

        # calc output; softmax over output weights dot hidden state
        y_t = T.flatten(T.nnet.softmax(T.dot(self.t_Wy, h_t)), 1)

        # return what we want to have per output step
        return [h_t, y_t]

    def t_y_softmax(self, t_x, t_h0):
        [_h_ts, t_y_softmax], _ = theano.scan(fn=self.recurrent_step,
                                             sequences=[t_x],
                                             outputs_info=[t_h0, None])
        return t_y_softmax
