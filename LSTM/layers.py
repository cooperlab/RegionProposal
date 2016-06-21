import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator

class GRULayer:

    def __init__(self, hidden_dim):

        # Assign variables
        self.hidden_dim = hidden_dim

        # Initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./(hidden_dim*hidden_dim)), np.sqrt(1./(hidden_dim*hidden_dim)), (3, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./(hidden_dim*hidden_dim)), np.sqrt(1./(hidden_dim*hidden_dim)), (3, hidden_dim, hidden_dim))
        b = np.zeros((3, hidden_dim, hidden_dim))

        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.learnable_params = [self.U, self.W, self.b]

        # SGD / rmsprop: Initialize parameters
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.cache_params = [self.mU, self.mW, self.mb]
        return

    def _forward_step(self, x_t, s_t_prev):

        # Define update operations
        z_t = T.nnet.hard_sigmoid(self.U[0].dot(x_t) + self.W[0].dot(s_t_prev) + self.b[0])
        r_t = T.nnet.hard_sigmoid(self.U[1].dot(x_t) + self.W[1].dot(s_t_prev) + self.b[1])
        c_t = T.nnet.sigmoid(self.U[2].dot(x_t) + self.W[2].dot(s_t_prev * r_t) + self.b[2])
        s_t = (T.ones_like(z_t) - z_t) * c_t + z_t * s_t_prev

        # In this case in particular
        x_t_next = s_t
        return [s_t, x_t_next]

class InnerProductLayer():

    def __init__(self, in_dim, out_dim):

        # Initialize the network parameters
        b = np.zeros((out_dim))
        W = np.random.uniform(-np.sqrt(1./(out_dim)), np.sqrt(1./(out_dim)), (in_dim, out_dim))

        # Theano: Created shared variables
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.learnable_params = [self.W, self.b]

        # SGD / rmsprop: Initialize parameters
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.cache_params = [self.mW, self.mb]
        return

    def _forward_step(self, x):
        return T.nnet.sigmoid(T.dot(x, self.W) + self.b)