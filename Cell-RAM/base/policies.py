from lasagne import init
import numpy as np
import theano.tensor as T
import theano
from models import CustomRecurrentModel
from theano.tensor.signal.pool import pool_2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class GaussianCRAMPolicy(CustomRecurrentModel):

    def __init__(self, specs):
        super(GaussianCRAMPolicy, self).__init__(specs)

        # Compute shapes
        out_shape_conv_1 = (self.input_shape[0], self.filter_shape_1[0],
                            (self.patch - self.filter_shape_1[2]) / self.stride[0] + 1,
                            (self.patch - self.filter_shape_1[3]) / self.stride[1] + 1)

        out_shape_pool_1 = (self.input_shape[0], self.filter_shape_1[0],
                            (int(out_shape_conv_1[2] / self.pooling_shape[0])),
                            (int(out_shape_conv_1[3] / self.pooling_shape[1])))

        out_shape_conv_2 = (self.input_shape[0], self.filter_shape_2[0],
                            (out_shape_pool_1[2] - self.filter_shape_2[2]) / self.stride[0] + 1,
                            (out_shape_pool_1[3] - self.filter_shape_2[3]) / self.stride[1] + 1)

        out_shape_pool_2 = (self.input_shape[0], self.filter_shape_2[0],
                            (int(out_shape_conv_2[2] / self.pooling_shape[0])),
                            (int(out_shape_conv_2[3] / self.pooling_shape[1])))

        fc_shape_1 = (np.prod(out_shape_pool_2[1:]), self.n_h_fc_1)
        out_shape_fc_1 = (self.input_shape[0], self.n_h_fc_1)

        fc_shape_2 = (self.n_h_fc_1, self.n_classes)
        out_shape_fc_2 = (self.input_shape[0], self.n_classes)

        # add params for core network
        # the number of hidden units is equal to the ouput of the first fully connected layer from the CNN
        #self.W_f_h_1 = self.add_param(init.GlorotNormal(), (self.n_h_fc_1, self.n_h_fc_1), name='W_f_h_1', type='r')
        #self.W_f_h_2 = self.add_param(init.GlorotNormal(), (self.n_h_fc_1, self.n_h_fc_1), name='W_f_h_2', type='r')
        #self.b_f_h = self.add_param(init.Constant(0.), (self.n_h_fc_1,), name='b_f_h', type='r')

        # add params for gated recurrent units
        # the number of hidden units is equal to the ouput of the first fully connected layer from the CNN
        self.W_gru_z = self.add_param(init.GlorotNormal(), (self.n_h_fc_1, self.n_h_fc_1), name='W_gru_z', type='r')
        self.U_gru_z = self.add_param(init.GlorotNormal(), (self.n_h_fc_1, self.n_h_fc_1), name='U_gru_z', type='r')
        self.b_gru_z = self.add_param(init.GlorotNormal(), (self.n_batch, self.n_h_fc_1), name='b_gru_z', type='r')

        self.W_gru_r = self.add_param(init.GlorotNormal(), (self.n_h_fc_1, self.n_h_fc_1), name='W_gru_r', type='r')
        self.U_gru_r = self.add_param(init.GlorotNormal(), (self.n_h_fc_1, self.n_h_fc_1), name='U_gru_r', type='r')
        self.b_gru_r = self.add_param(init.GlorotNormal(), (self.n_batch, self.n_h_fc_1), name='b_gru_r', type='r')

        self.W_gru_h = self.add_param(init.GlorotNormal(), (self.n_h_fc_1, self.n_h_fc_1), name='W_gru_h', type='r')
        self.U_gru_h = self.add_param(init.GlorotNormal(), (self.n_h_fc_1, self.n_h_fc_1), name='U_gru_h', type='r')
        self.b_gru_h = self.add_param(init.GlorotNormal(), (self.n_batch, self.n_h_fc_1), name='b_gru_h', type='r')

        # add params for action network (location)
        self.W_f_l = self.add_param(init.GlorotNormal(), (self.n_h_fc_1, 2), name='W_f_l', type='r')
        self.b_f_l = self.add_param(init.Constant(0.), (2,), name='b_f_l', type='r')

        # add params for action network (classification)
        self.W_conv_1 = self.add_param(init.GlorotNormal(), self.filter_shape_1, name='W_conv_1', type='p')
        self.b_conv_1 = self.add_param(init.Constant(0.), out_shape_conv_1, name='b_conv_1', type='p')
        self.W_conv_2 = self.add_param(init.GlorotNormal(), self.filter_shape_2, name='W_conv_2', type='p')
        self.b_conv_2 = self.add_param(init.Constant(0.), out_shape_conv_2, name='b_conv_2', type='p')
        self.W_fc_1 = self.add_param(init.GlorotNormal(), fc_shape_1, name='W_fc_1', type='p')
        self.b_fc_1 = self.add_param(init.Constant(0.), out_shape_fc_1, name='b_fc_1', type='p')
        self.W_fc_2 = self.add_param(init.GlorotNormal(), fc_shape_2, name='W_fc_2', type='p')
        self.b_fc_2 = self.add_param(init.Constant(0.), out_shape_fc_2, name='b_fc_2', type='p')

        self._srng = RandomStreams(np.random.randint(1, 2147462579))
        return

    def step_forward(self):

        # Helper function called by scan
        def _step(g_noise, loc_tm1, h_tm1, x):
            """
            :param loc_tm1:
            :param h_tm1:
            :param x:
            :return: location action stored in loc_tm1 which belongs to the interval [-1,1]
            Obs: This function works only for n_batch = 1
            """

            def _foward_pred(x_t):
                """
                This is a helper-function for computing the prediction action.
                :param x_t:
                :return: probabilities for each class (n_batch, n_classes)
                """

                # Convolution 1
                conv_out_1 = T.nnet.relu(T.nnet.conv2d(x_t, self.W_conv_1, subsample=self.stride) + self.b_conv_1)
                pool_out_1 = pool_2d(conv_out_1, self.pooling_shape, ignore_border=True)

                # Convolution 2
                conv_out_2 = T.nnet.relu(
                    T.nnet.conv2d(pool_out_1, self.W_conv_2, subsample=self.stride) + self.b_conv_2)
                pool_out_2 = pool_2d(conv_out_2, self.pooling_shape, ignore_border=True)

                # Fully Connected
                fc_out_1 = T.nnet.relu(
                    T.dot(T.unbroadcast(pool_out_2.reshape((self.n_batch, -1)), 0), self.W_fc_1) + self.b_fc_1)
                return T.nnet.softmax(T.dot(fc_out_1, self.W_fc_2) + self.b_fc_2)

            # Helper function
            def _rho(loc_tm1, x):
                """
                This function will be replaced
                :param loc_tm1:
                :param x:
                :return: x_t (n_batch, channels, img_height, img_width) and x_flattened (n_batch, img_height*img_width)
                """

                x_t_i = []
                loc_tm1_clipped = T.clip(loc_tm1, -1, 1)
                for b in xrange(self.n_batch):
                    y_start = theano.tensor.cast((1 + loc_tm1_clipped[0][0]) * (self.img_height - self.patch) / 2,
                                                 'int32')
                    x_start = theano.tensor.cast((1 + loc_tm1_clipped[0][1]) * (self.img_width - self.patch) / 2,
                                                 'int32')
                    img = x[b, :, y_start:y_start + self.patch, x_start:x_start + self.patch]
                    x_t_i.append(img)

                x_t = T.stack(x_t_i, axis=0)
                return x_t

            # Helper function -- This function extract features from the predicted window using the CNN from the classifier
            def _f_g(x_t, loc_tm1):

                # Convolution 1
                conv_out_1 = T.nnet.relu(T.nnet.conv2d(x_t, self.W_conv_1, subsample=self.stride) + self.b_conv_1)
                pool_out_1 = pool_2d(conv_out_1, self.pooling_shape, ignore_border=True)

                # Convolution 2
                conv_out_2 = T.nnet.relu(
                    T.nnet.conv2d(pool_out_1, self.W_conv_2, subsample=self.stride) + self.b_conv_2)
                pool_out_2 =  pool_2d(conv_out_2, self.pooling_shape, ignore_border=True)

                # Fully Connected
                fc_out_1 = T.nnet.relu(
                    T.dot(T.unbroadcast(pool_out_2.reshape((self.n_batch, -1)), 0), self.W_fc_1) + self.b_fc_1)
                return fc_out_1

            # Helper function -- This function updates the hidden state units
            def _f_h(h_tm1, x_in):

                z =  T.nnet.hard_sigmoid(T.dot(x_in, self.U_gru_z) + T.dot(h_tm1, self.W_gru_z) + self.b_gru_z)
                r =  T.nnet.hard_sigmoid(T.dot(x_in, self.U_gru_r) + T.dot(h_tm1, self.W_gru_r) + self.b_gru_r)
                h =  T.tanh(T.dot(x_in, self.U_gru_h) + T.dot((h_tm1 * r), self.W_gru_h) + self.b_gru_h)
		return (1 - z) * h + z * h_tm1 
                #return T.nnet.relu(T.dot(h_tm1, self.W_f_h_1) + T.dot(g_t, self.W_f_h_2) + self.b_f_h.dimshuffle('x', 0))

            # Helper function
            def _f_l(h_t):
                return T.dot(h_t, self.W_f_l) + self.b_f_l.dimshuffle('x', 0)

            x_t = _rho(loc_tm1, x)
            g_t = _f_g(x_t, loc_tm1)
            h_t = _f_h(h_tm1, g_t)

            # Stores the predicted means (n_batch, 2)
            loc_mean_t = _f_l(h_t)

            # Sample from a normal distribution for each batch
            #loc_t = T.unbroadcast(T.stack([self._srng.normal((2, ), avg=loc_mean_t[b, :], std=self.sigma) for b in xrange(self.n_batch)]), 0)
            loc_t = theano.gradient.zero_grad(loc_mean_t + g_noise)

            # Get prediction
            p_t = _foward_pred(x_t)
            return loc_mean_t, loc_t, h_t, x_t, p_t

        # Propagate trough time
        [loc_means_t, locs_t, h_ts, x_ts, p_ts], updates = theano.scan(_step,
                                                 outputs_info=[None,
                                                               T.unbroadcast(theano.shared(np.random.uniform(-1.0, 1.0, (self.n_batch, 2)).astype('float32')),0),
                                                               T.unbroadcast(theano.shared(np.random.uniform(-1.0, 1.0,(self.n_batch, self.n_h_fc_1)).astype('float32')), 0),
                                                               None,
                                                               None],
                                                 sequences=[self._srng.normal((self.n_steps, self.n_batch, 2), avg=0.0, std=self.sigma)],
                                                 non_sequences=[self.in_var],
                                                 n_steps=self.n_steps)

        # Swap first two dim
        loc_means_t = loc_means_t.dimshuffle(1, 0, *range(2, loc_means_t.ndim))
        locs_t = locs_t.dimshuffle(1, 0, *range(2, locs_t.ndim))
        h_ts = h_ts.dimshuffle(1, 0, *range(2, h_ts.ndim))
        x_ts = x_ts.dimshuffle(1, 0, *range(2, x_ts.ndim))
        p_ts = p_ts.dimshuffle(1, 0, *range(2, p_ts.ndim))
        return loc_means_t, locs_t, h_ts, x_ts, p_ts
