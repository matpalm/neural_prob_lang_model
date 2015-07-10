#!/usr/bin/env python
import numpy as np
import util
import theano
import theano.tensor as T

class BidirectionalRnn(object):
    def __init__(self, n_in, n_hidden):
        self.t_Wx_f = util.sharedMatrix(n_hidden, n_in, 'Wx_f')
        self.t_Wrec_f = util.sharedMatrix(n_hidden, n_hidden, 'Wrec_f')
        self.t_Wy_f = util.sharedMatrix(n_in, n_hidden, 'Wy_f')
        self.t_Wx_b = util.sharedMatrix(n_hidden, n_in, 'Wx_b')
        self.t_Wrec_b = util.sharedMatrix(n_hidden, n_hidden, 'Wrec_b')
        self.t_Wy_b = util.sharedMatrix(n_in, n_hidden, 'Wy_b')

    def params(self):
        return [self.t_Wx_f, self.t_Wrec_f, self.t_Wy_f, self.t_Wx_b, self.t_Wrec_b, self.t_Wy_b]

    def scan_through_x(self, 
                       x_t,            # sequence to scan
                       h_t0,           # recurrent state
                       Wx, Wrec, Wy):  # non_sequences
        # calc new hidden state; elementwise add of embedded input &
        # recurrent weights dot _last_ hiddenstate
        h_t = T.nnet.sigmoid(Wx[:, x_t] + T.dot(Wrec, h_t0))
        # calc contribution to y
        y_t = T.dot(Wy, h_t)
        # return next hidden state and y contribution
        return [h_t, y_t]

    def t_y_softmax(self, t_x, t_h0):
        # forward scan through x collecting contributions to y
        [_h_ts, y_ts_f], _ = theano.scan(fn = self.scan_through_x,
                                         go_backwards = False,
                                         sequences = [t_x],
                                         non_sequences = [self.t_Wx_f, self.t_Wrec_f, self.t_Wy_f],
                                         outputs_info = [t_h0, None])
        # backwards scan through x collecting contributions to y
        [_h_ts, y_ts_b], _ = theano.scan(fn = self.scan_through_x,
                                         go_backwards = True,
                                         sequences = [t_x],
                                         non_sequences = [self.t_Wx_b, self.t_Wrec_b, self.t_Wy_b],
                                         outputs_info = [t_h0, None])
        # elementwise combine y contributions and apply softmax
        t_y_softmax, _ = theano.scan(fn = lambda y_f, y_b: T.flatten(T.nnet.softmax(y_f + y_b), 1),
                                     sequences = [y_ts_f, y_ts_b],
                                     outputs_info = [None])
        return t_y_softmax
