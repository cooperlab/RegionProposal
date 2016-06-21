import theano.tensor as T
import theano
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

class RecurrentNeuralNetwork:

    def __init__(self, layers, pred_layers, patience=20, btt=2):

        # Define layers
        self.layers = layers
        self.pred_layers = pred_layers
        self.patience_max = patience
        self.patience = patience
        self.best_score = 0
        self.btt = btt

        # We store the Theano graph here
        self.__theano_build__()

    def __theano_build__(self):

        # Define io variables
        im = T.dmatrix('im')
        gt = T.dmatrix('gt')

        def _forward(im, s_ti):

            # Forward step
            x_ts = [s_ti[-1] * im] # Verificar se eh bom usar a sigmoid aqui
            s_ti_next = []
            for k, layer in enumerate(self.layers):
                [s, x] = layer._forward_step(x_ts[-1], s_ti[k])
                x_ts.append(x)
                s_ti_next.append(s)

            # Compute output
            o_t = x_ts[-1]
            return [o_t, s_ti_next]

        def _forward_tt(im):

            # Initialize states
            s_tti = [[np.random.uniform(0.0, 1.0, (layer.hidden_dim, layer.hidden_dim)) for layer in self.layers]] # Verificar a inicializacao dos estados
            #s_tti = [[T.ones((layer.hidden_dim, layer.hidden_dim)) for layer in self.layers]] # Verificar a inicializacao dos estados

            outs = []
            for k in range(self.btt):
                o, s_ti = _forward(im, s_tti[-1])
                outs.append(o)
                s_tti.append(s_ti)
            return [outs[-1], s_tti[-1]]

        def _forward_pred(in_data):
            in_next = [in_data]
            for layer in self.pred_layers:
                out = layer._forward_step(in_next[-1])
                in_next.append(out)
            return in_next[-1]

        def _histogram(x, nbin=20, xlim=(0, 1)):

            y = T.floor((x - xlim[0]) * nbin / (xlim[1] - xlim[0]))
            hist = T.stack([T.cast(T.eq(y, b), T.config.floatX).sum() for b in range(nbin)])
            return hist/hist.sum()

        # Create theano graph
        [proposed_region, s] = _forward_tt(im)
        #mask = T.switch(T.lt(proposed_region, 0.5), 0.0, 1.0)
        mask = proposed_region
        applied_mask = mask * im
        #bow = _histogram(applied_mask, 256)
        o = _forward_pred(applied_mask.flatten())
        cost = self.compute_loss(gt, o, im)

        # SGD parameters
        self.learning_rate = T.scalar('learning_rate')
        self.decay = T.scalar('decay')

        # Assign functions
        #self.get_histogram = theano.function([im], bow)
        self.get_mask = theano.function([im], mask)
        self.propose_region = theano.function([im], applied_mask)
        self.predict = theano.function([im], o)
        self.ce_error = theano.function([im, gt], cost)
        self.sgd_step = theano.function(
            [im, gt, self.learning_rate, theano.Param(self.decay, default=0.9)],
            updates=[(par, update) for par, update in zip(self.params+self.m_params, self.updates(cost)+self.m_updates(cost))])
        return

    def reset_patience(self):
        self.patience = self.patience_max
        return

    def decrease_patience(self):
        self.patience -= 1
        return

    @property
    def params(self):
        return [par for l in self.layers for par in l.learnable_params] + [par for l in self.pred_layers for par in l.learnable_params]

    @property
    def m_params(self):
        return [m_par for l in self.layers for m_par in l.cache_params] + [m_par for l in self.pred_layers for m_par in l.cache_params]

    def grads(self, cost):
        #return [T.clip(T.grad(cost, p), -1, 1) for p in self.params]
        return [T.grad(cost, p) for p in self.params]

    def m_updates(self, cost):
        return [self.decay * m_p + (1 - self.decay) * g ** 2 for (m_p, g) in zip(self.m_params, self.grads(cost))]

    def updates(self, cost):
        return [p - self.learning_rate * g / T.sqrt(m_p + 1e-6) for (p, g, m_p) in zip(self.params, self.grads(cost), self.m_updates(cost))]

    def compute_loss(self, target, out, im):
        # Region is represented as an matrix NxN with elements in a range from 0 to 1.
        return T.nnet.binary_crossentropy(out, target).mean()

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x, [[y]]) for x, y in zip(X, Y)])/float(len(X))

    def save(self, fname):
        return np.savez(fname, params=self.params, m_params=self.m_params)

    def load(self, fname):
        npzfile=np.load(fname)
        params = npzfile['params']
        m_params = npzfile['m_params']
        for par_saved, par in zip(params, self.params):
            par.set_value(par_saved.get_value())
        for m_par_saved, m_par in zip(m_params, self.m_params):
            m_par.set_value(m_par_saved.get_value())
        return

class MultilayerPerceptron:

    def __init__(self, layers, patience=20):

        # Define layers
        self.layers = layers
        self.patience_max = patience
        self.patience = patience
        self.best_score = 0

        # We store the Theano graph here
        self.__theano_build__()

    def __theano_build__(self):

        # Define io variables
        im = T.dmatrix('im')
        gt = T.dmatrix('gt')

        def _forward(im):

            # Forward step
            x_ts = [im] # Verificar se eh bom usar a sigmoid aqui
            for k, layer in enumerate(self.layers):
                x = layer._forward_step(x_ts[-1])
                x_ts.append(x)

            # Compute output
            o_t = x_ts[-1]
            return o_t

        # Create theano graph
        o = _forward(im)
        cost = self.compute_loss(gt, o)

        # SGD parameters
        self.learning_rate = T.scalar('learning_rate')
        self.decay = T.scalar('decay')

        # Assign functions
        self.predict = theano.function([im], o)
        self.ce_error = theano.function([im, gt], cost)
        self.sgd_step = theano.function(
            [im, gt, self.learning_rate, theano.Param(self.decay, default=0.9)],
            updates=[(par, update) for par, update in zip(self.params+self.m_params, self.updates(cost)+self.m_updates(cost))])
        return

    def reset_patience(self):
        self.patience = self.patience_max
        return

    def decrease_patience(self):
        self.patience -= 1
        return

    @property
    def params(self):
        return [par for l in self.layers for par in l.learnable_params]

    @property
    def m_params(self):
        return [m_par for l in self.layers for m_par in l.cache_params]

    def grads(self, cost):
        #return [T.clip(T.grad(cost, p), -1, 1) for p in self.params]
        return [T.grad(cost, p) for p in self.params]

    def m_updates(self, cost):
        return [self.decay * m_p + (1 - self.decay) * g ** 2 for (m_p, g) in zip(self.m_params, self.grads(cost))]

    def updates(self, cost):
        return [p - self.learning_rate * g / T.sqrt(m_p + 1e-6) for (p, g, m_p) in zip(self.params, self.grads(cost), self.m_updates(cost))]

    def compute_loss(self, target, out):
        # Region is represented as an matrix NxN with elements in a range from 0 to 1.
        print target, out
        return T.nnet.binary_crossentropy(out, target).mean()

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error([x], [[y]]) for x, y in zip(X, Y)])/float(len(X))

    def save(self, fname):
        return np.savez(fname, params=self.params, m_params=self.m_params)

    def load(self, fname):
        npzfile=np.load(fname)
        params = npzfile['params']
        m_params = npzfile['m_params']
        for par_saved, par in zip(params, self.params):
            par.set_value(par_saved.get_value())
        for m_par_saved, m_par in zip(m_params, self.m_params):
            m_par.set_value(m_par_saved.get_value())
        return