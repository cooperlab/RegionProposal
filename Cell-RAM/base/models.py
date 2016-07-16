import numpy as np
import theano.tensor as T
from lasagne.utils import create_param

class CustomRecurrentModel(object):

    def __init__(self, specs):
        """
        :param input_shape:
        :param patience:
        :param learning_rate:
        """
        # Define params
        self.in_var = T.tensor4(name='input', dtype=T.config.floatX)
        self.out_var = T.ivector(name='output')
        self.input_shape = specs['input_shape']
        self.patience_max = specs['patience']
        self.patience = specs['patience']
        self.best_score = 0
        self.learning_rate = specs['learning_rate']
        self.patch = specs['patch']
        self.n_steps = specs['n_steps']
        self.n_batch = self.input_shape[0]
        self.n_channels = self.input_shape[1]
        self.img_height = self.input_shape[2]
        self.img_width = self.input_shape[3]
        self.params = []
        self.r_params = []
        self.p_params = []
        self.n_h_g = specs['n_h_g']
        self.n_h_l = specs['n_h_l']
        self.n_f_g = specs['n_f_g']
        self.n_f_h = specs['n_f_h']
        self.filter_shape_1 = specs['filter_shape_1']
        self.filter_shape_2 = specs['filter_shape_2']
        self.stride = specs['stride']
        self.pooling_shape = specs['pooling_shape']
        self.n_h_fc_1 = specs['n_h_fc_1']
        self.n_classes = specs['n_classes']
        self.sigma = specs['sigma']
        self.N = specs['n_trials']
        return

    def add_param(self, init, in_shape, name, type):
        par = create_param(init, in_shape, name)
        self.params.append(par)
        if type == 'r':
            self.r_params.append(par)
        elif type == 'p':
            self.p_params.append(par)
        return par

    def get_all_param_values(self):
        return [p.get_value() for p in self.params]

    def set_all_param_values(self, values):
        return [p.set_value(v) for p, v in zip(self.params, values)]

    def get_all_params(self):
        return self.params

    def reset_patience(self):
        self.patience = self.patience_max
        return

    def decrease_patience(self):
        self.patience -= 1
        return

    def save(self, fname):
        return np.savez(fname, values=self.get_all_param_values())

    def load(self, fname):
        npzfile = np.load(fname)
        values = npzfile['values']
        self.set_all_param_values(values)
        return