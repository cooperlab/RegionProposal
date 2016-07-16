import theano
import theano.tensor as T
import numpy as np
import lasagne
from base.policies import GaussianCRAMPolicy
from theano.tensor.extra_ops import cumsum
from theano.compile.nanguardmode import NanGuardMode
import sys

class CRAM():
    def __init__(self, specs):

        # Define policy
        self.policy = GaussianCRAMPolicy(specs)

        # Compile theano graph
        self._theano_build = self._theano_build()

    def _theano_build(self):
        """
        :return:
        Obs: p_ts is (n_batch, T, 2)
        """
        # Step forward
        loc_means_t, locs_t, h_ts, x_ts, p_ts = self.policy.step_forward()

        # Compute prediction for time stamp T
        # Obs: just the last prediction is being considered for computing the loss and updates
        # and its size becomes (n_batch, 1)
        prob_T = p_ts[:, -1, :]
        preds_T = T.argmax(prob_T, axis=1)

        # This variable stores the prediction for an entire path
        preds = T.argmax(p_ts, axis=2)

        # Compute loss and updates
        r_loss = self._r_loss(p_ts, self.policy.out_var)
        p_loss = self._p_loss(prob_T, self.policy.out_var)
        updates = self._updates(prob_T, self.policy.out_var)

        # Build theano functions
        self.fit = theano.function(inputs=[self.policy.in_var, self.policy.out_var], updates= updates)
        self.compute_p_loss = theano.function(inputs=[self.policy.in_var, self.policy.out_var], outputs=[p_loss])
        self.compute_r_loss = theano.function(inputs=[self.policy.in_var, self.policy.out_var], outputs=[r_loss])
        self.proba = theano.function(inputs=[self.policy.in_var], outputs=[prob_T])
        self.predict = theano.function(inputs=[self.policy.in_var], outputs=[preds_T])
        self.propose_region = theano.function(inputs=[self.policy.in_var], outputs=[locs_t])
        self.debug = theano.function(inputs=[self.policy.in_var, self.policy.out_var], outputs=[r_loss], on_unused_input='ignore')
        #self.debug = theano.function(inputs=[self.policy.in_var, self.policy.out_var], outputs=[self._r_grads(self.policy.out_var)])
        return

    def _updates(self, prob, y):
        """
        This function compute the updates using ADAM
        :param y:
        :return:
        Obs: the prediction loss considers just the prediction from the last time stamp
        """
        #return lasagne.updates.adam(loss_or_grads=self._r_grads(y) + self._p_grads(self._p_loss(prob, y)),
        #                            params=self.policy.r_params + self.policy.p_params,
        #                            learning_rate=self.policy.learning_rate)
        return lasagne.updates.adam(loss_or_grads= self._r_grads(y) + self._p_grads(self._p_loss(prob, y)),
                                    params=self.policy.r_params + self.policy.p_params,
                                    learning_rate=self.policy.learning_rate)

    def _acc_score(self, pred, y):
        """
        :param pred:
        :param y:
        :return:
        """
        return T.eq(pred, y)

    def _r_loss(self, probs, y):
        """
        :param preds: (n_batch, T) this variable stores the predictions for one path of size T
        :param y:   (n_batch, ) this variable stores the targets
        :return:
        """
        # y_rep: (n_batch, T, )
        y_rep = T.stack([T.fill(T.zeros((self.policy.n_steps)), y[b]) for b in xrange(self.policy.n_batch)], axis=0)
        return T.nnet.binary_crossentropy(probs[:, :, 1], y_rep).mean(axis=[0,1])
        #return T.neq(preds, y_rep).mean(axis=[0,1])

    def _p_loss(self, prob, y):
        """
        :param prob: (n_batch, 2)
        :param y: (n_batch, )
        :return:
        Obs: This function is considering just a binary classification problem
        """
        return T.nnet.binary_crossentropy(prob[:,1], y).mean()

    def _log_likelihood(self, x_vars, means):
        """
        This function computes the symbolic log-likelihood for a diagonal gaussian defined by the given
        means and a fixed sigma.
        :param x_vars:
        :param means:
        :return:
        """
        std = T.fill(T.zeros_like(means), self.policy.sigma)
        zs = (x_vars - means)/std
        return -T.sum(T.log(std), axis=-1)\
               -0.5 * T.sum(T.square(zs), axis=-1)\
               -0.5 * means.shape[-1] * np.log(2 * np.pi)

    def _collect_samples(self, y):
        """
        This function collect N samples of size T using the current policy.
        :param y:
        :return: locations (n_batch, N, T, 2), probabilities (n_batch, N, T, n_classes),
        rewards (n_batch, N, T, ) and returns (n_batch, N, T, )
        """
        means = [];locs = [];probs = [];returns = [];preds=[]

        # Reshape target labels to match the classification outputs along each path of length T
        y_rep = T.stack([T.fill(T.zeros((self.policy.n_steps)), y[b]) for b in xrange(self.policy.n_batch)], axis=0)
        for _ in xrange(self.policy.N):
            loc_means_t, locs_t, _, x_ts, p_ts = self.policy.step_forward()
            locs.append(locs_t)
            means.append(loc_means_t)
            probs.append(p_ts)
            pred = np.argmax(p_ts, axis=2)
            preds.append(pred)
            rewards = self._acc_score(pred, y_rep)
            returns.append(cumsum(rewards, axis=1))

        locs = T.stack(locs).dimshuffle(1, 0, *range(2, T.stack(locs).ndim))
        means = T.stack(means).dimshuffle(1, 0, *range(2, T.stack(means).ndim))
        preds = T.stack(preds).dimshuffle(1, 0, *range(2, T.stack(preds).ndim))
        returns = T.stack(returns).dimshuffle(1, 0, *range(2, T.stack(returns).ndim))

        return locs, means, preds, returns

    def _r_grads(self, y):
        """
        :param r:
        :param loc_mean_t:
        :return:
        baseline ref: Weaver, Lex; Tao, Nigel; The optimal reward baseline for gradient-based reinforcement learning.
        """

        # TODO: Incluir desconto

        # Get N sequences of size T
        locs, means, pred, returns = self._collect_samples(y)

        # Get log-likelihood (n_batch, N, T)
        log_pi = self._log_likelihood(locs, means)

        # Get policy params only
        r_params = self.policy.r_params

        # Get sum over T
        # log_pi_over_T: (n_batch, N)
        # returns_over_T: (n_batch, N)
        log_pi_over_T = T.sum(log_pi, axis=2)
        returns_over_T = T.sum(returns, axis=2)

        # TODO: Optimize these loops
        # For each param
        # Set b = 0
        # Obs: this code considers only batch size of 1!
        b = 0

        # Compute gradients for each sample wrt each param
        jacobian_over_T = [[T.grad(log_pi_over_T[b, n], r_params[p]) for p in xrange(len(r_params))] for n in xrange(self.policy.N)]

        # estimated_grads: (n_params)
        estimated_grads = []
        for p in xrange(len(r_params)):

            # Get baseline
            numerator = T.zeros_like(r_params[p])
            denominator = T.zeros_like(r_params[p])
            for n in xrange(self.policy.N):
                numerator += returns_over_T[b, n] * (jacobian_over_T[n][p] ** 2)
                denominator += (jacobian_over_T[n][p] ** 2)
            b_p = numerator/(denominator + 1e-8)

            # Estimate gradient
            grad = T.zeros_like(r_params[p])
            for n in xrange(self.policy.N):

                # The mean is taken over the N samples collected
                advantages = (returns_over_T[b, n] - b_p)

                # Whitening the advantages
                #advantages = (advantages - T.mean(advantages))/(T.std(advantages)+ 1e-8)

                grad += (1.0 / float(self.policy.N)) * jacobian_over_T[n][p] * advantages

            estimated_grads.append(grad.astype('float32'))
        return estimated_grads

    def _p_grads(self, loss):
        """
        :param loss:
        :return:
        """
        return T.grad(loss, self.policy.p_params, disconnected_inputs='ignore')

    def calculate_total_loss(self, X_val, y_val):
        """
        In this function we consider the proposal loss and the prediction loss for performing validation.
        :param X_val:
        :param y_val:
        :return:
        """
        return np.mean([float(self.compute_p_loss(np.reshape(x, self.policy.input_shape), [y])[0]) + float(self.compute_r_loss(np.reshape(x, self.policy.input_shape), [y])[0])
                        for x, y in zip(X_val, y_val)])

    def calculate_proposal_loss(self, X_val, y_val):
        """
        :param X_val:
        :param y_val:
        :return:
        """
        return np.mean([float(self.compute_r_loss(np.reshape(x, self.policy.input_shape), [y])[0]) for x, y in zip(X_val, y_val)])

    def init_score(self, score):
        self.policy.best_score = score

    def is_finished(self):
        if self.policy.patience == 0:
            self.policy.load('trained_model.npz')
            return True
        return False

    def get_score(self):
        return self.policy.best_score

    def update_score(self, score):
        if score < self.policy.best_score:

            print('Saving model to trained_model.npz')
            # Better score found
            self.policy.save('trained_model.npz')
            self.policy.reset_patience()
            self.policy.best_score = score

        else:
            self.policy.decrease_patience()
            print('Patience:')
            print(self.policy.patience)

        print("\n")
        sys.stdout.flush()

    def load(self, fname):
        self.policy.load(fname)
        return

## Attemp to remove loop - didn't work
## Helper-function
# def _estimate_grad(n):

# Compute gradients for each sample wrt each param
# grad_log_pi_over_T = T.grad(log_pi_over_T[b, n], r_params[p], disconnected_inputs='ignore')
# tmp_grads.append(grad_log_pi_over_T)

# Compute optimal baseline
# baseline = ((T.square(grad_log_pi_over_T)) * returns_over_T[b, n])/(T.square(grad_log_pi_over_T))

# The mean is taken over the N samples collected
# grad = (1.0 / float(self.policy.N * self.policy.n_batch))* grad_log_pi_over_T * returns_over_T[b, n]
# return grad.astype('float32')

# fragmented_grads, updates = theano.scan(_estimate_grad, outputs_info=[None], sequences=[T.arange(self.policy.N)])
