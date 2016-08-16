# Region Proposal Project
## Summary
The ultimate goal of this project is exploring high and low-resolution images techniques for performing
high-level inferences over complete tissue medical slides, which presents a few challenging aspects regarding
storage and processing. The high-level inferences consist of learning patterns between summarized slides and
target labels for solving a specific problem. For example, it could be interesting to calculate the abundance (by
area) of regions like angiogenesis, necrosis, and infiltrating lymphocytes to infer patient-level characteristics.
A tissue slide can easily reach around 20GB in-memory, which turns intractable many approaches of feeding
those slides entirely to a classifier. One recent method for addressing this problem is sampling regions based
on low-resolution features (i.e. texture, boundaries) for extracting information about the tissue. Moreover, this
method can also help with speeding up the processing, since after cropping the proposed regions, the problem
would be turned into a tractable learning problem.

##Tutorials
This section is divided in two parts. First, we implemented a soft-attention model utilizing Long Short Term Memories 
in order to propose regions at pixel level for classification similarly as in (ref). The second approach consists of using 
Reinforcement Learning for predicting fixed sized windows for feeding a classifier composed by Convolutional Neural Networks
similarly as in (ref).

Part A - Soft-Attention Model using LSTM
** TODO

Part B - Hard-Attention Model using RL (Cell-RAM)
** TODO

#API Documentation

The Cell Recurrent Attention Model (Cell-RAM) API is implemented using Theano XXX and provides
tools for defining and training a recurrent model using the algorithm REINFORCE (ref).

```
base.models.CustomRecurrentModel(object)
```
This class defines the hyperparameters of a recurrent model and implements the basic 
functions for building up a custom recurrent model.

```
base.models.CustomRecurrentModel.__init__(specs)
```
This function initializes a object for the recurrent model with parameters defined in specs.

> ####Parameters: 
> > - specs - set of parameters that defines a model.


```
base.models.CustomRecurrentModel.add_param(init, in_shape, name, type)
```
This function returns a theano variable initialized as determined.

> ####Parameters: 
> > - init     - param initialization from lasagne
> > - in_shape - param shape
> > - name     - param name
> > - type     - determines if the param is used in the prediction phase 'p' or reinforce phase 'r'

```
base.models.CustomRecurrentModel.get_all_param_values()
```
This function returns all parameters values defined for the model and is used for saving/loading a model.

```
base.models.CustomRecurrentModel.set_all_param_values(values)
```
This function sets the parameters of a given model to the given values.

> ####Parameters: 
> > - values - consists of a set of values for each parameter of the model

```
base.models.CustomRecurrentModel.get_all_params()
```
This function returns the value's container for each parameter.

```
base.models.CustomRecurrentModel.reset_patience()
```
This function reset the patience parameter used in the early stopping implementation.

```
base.models.CustomRecurrentModel.decrease_patience()
```
This function decrease the patience parameter used in the early stopping implementation.

```
base.models.CustomRecurrentModel.save(fname)
```
This function saves the model to file.

> ####Parameters: 
> > - fname - file name 

```
base.models.CustomRecurrentModel.load(fname)
```
This function loads the model from a file.

> ####Parameters: 
> > - fname - file name

```
base.GaussianCRAMPolicy(CustomRecurrentModel)
```
This class defines a policy (RNN) and the forward learning steps.

```
base.GaussianCRAMPolicy.__init__(specs)
```
This function initializes the policy with the parameters given in specs and 
defines the learnable parameters for performing classification, updating the hidden state and
predicting the location of the windows.

> ####Parameters: 
> - specs - set of parameters
	
```
base.GaussianCRAMPolicy.step_forward()
```
This function performs the forward propagation through-time using theano scan and initializes
the hidden state and window's location.

```
base.GaussianCRAMPolicy.step_forward._step(g_noise, loc_tm1, h_tm1, x)
```
This is a helper function that receives a gaussian noise, previous location and hidden state, entire image and returns
the next values for the location and hidden state. It extracts the proposed region given a window's location, updates the hidden units, predict a new window's location
and performs classification on the proposed region in this order.

> ####Parameters: 
> > - g_noise - samples of a gaussian noise used for sampling the predicted distribution for the location
> > - loc_tm1 - previous location
> > - h_tm1   - previous hidden state
> > - x       - entire image

```
base.GaussianCRAMPolicy.step_forward._step._foward_pred(x_t)
```
This function performs classification using a convolutional neural networks which is trained along with the reinforcement learning step
and returns the probabilities for each class corresponding with the given samples.

> ####Parameters: 
> > - x_t - proposed region at time-stamp t
	
```
base.GaussianCRAMPolicy.step_forward._step._rho(loc_tm1, x)
```
This function crops the entire image accordingly to the previous predicted location
and returns a stack of cropped images one for each batch.

> ####Parameters:
> > - loc_tm1 - previous location
> > - x       - entire image     

```
base.GaussianCRAMPolicy.step_forward._step._f_g(x_t)
```
This function extracts features from the proposed region using the same CNN from the classification step, which means that
the feature extraction of the proposed region and the classification phase shares the same parameters. In this step
the activations right before the softmax are used as features for updating the hidden units.

> ####Parameters: 
> > - x_t - proposed region at time-stamp t
	
```
base.GaussianCRAMPolicy.step_forward._step._f_h(h_tm1, x_in)
```
This function updates the hidden state using gated recurrent units.

> ####Parameters: 
> > - h_tm1 - previous hidden state
> > - x_in  - extracted features from the proposed region

```
base.GaussianCRAMPolicy.step_forward._step._f_l(h_t)
```
This function performs a dot product for estimating the next locations for the window given the current hidden state

> ####Parameters:
> > - h_t - current hidden state

```
CRAM()
```
This class abstract the world of our problem which includes the policy, agent and the learning routines.

```
CRAM.__init__(specs)
```
This function initializes a instance of CRAM using the defined parameters passed in specs.

> ####Parameters: 
> > - specs - set of parameters

```
CRAM._theano_build()
```
This function define the theano graphs for all theano functions used in the model.

```
CRAM._updates(prob, y)
```
This function returns the updates for the learnable parameters during the prediction step using ADAM.

> ####Parameters: 
> > - prob - probabilities for each class for a given mini-batch
> > - y    - ground-truth for a given mini-batch

```				  
CRAM._acc_score(pred, y)
```
This functions returns the accuracy for a given mini-batch.

> ####Parameters: 
> > - pred - predicted classes for a giben mini-batch
> > - y    - ground-truth for a given mini-batch

```				  				  
CRAM._r_loss(preds, y)
```
This function computes the loss considered during the reinforce learning step.

> ####Parameters: 
> > - preds - predicted classes for a given mini-batch
> > - y     - ground-truth for a given mini-batch

```				 
CRAM._p_loss(prob, y)
```
This function returns the loss used for training the classifier.

> ####Parameters:
> > - prob - probabilites for each class given a mini-batch
> > - y    - ground-truth for a given mini-batch

```				  
CRAM._log_likelihood(x_vars, means)
```
This function returns the log-likelihood.

> ####Parameters: 
> > - x_vars - samples from the predicted distribution
> > - means  - means of the predicted distribution

```
CRAM._collect_samples(y)
```
This function collects samples from the environment.

> ####Parameters:
> > - y - ground-truth

```
CRAM._r_grads(y)
```
This function returns the estimated gradients for the REINFORCE algorithm.

> ####Parameters:
> > - y - ground-truth

```	
CRAM._p_grads(loss)
```
This function returns the gradients for the parameters in the classifier.

> ####Parameters: 
> > - y - ground-truth

```	
CRAM.calculate_total_loss(X_val, y_val)
```
This function computes the total loss for being used during validation step regarding the classification.

> ####Parameters:  
> > - X_val - validation samples
> > - y_val - validation ground-truth

```
CRAM.calculate_proposal_loss(X_val, y_val)
```
This function computes the total loss for being used during validation step regarding th region proposal.

> ####Parameters:  

> > - X_val - validation samples
> > - y_val - validation ground-truth

```
CRAM.init_score(score)
```
This function initialize the score for the Early Stopping implementation.

> ####Parameters:  

> > - score - measurement of quality for our model 

```
CRAM.is_finished()
```
This function checks if the model should stop or not.

```
CRAM.get_score()
```
This function returns the current best score achieved during learning.

```
CRAM.update_score(score)	
```
This function updates the score when a better score is found.

> ####Parameters:  

> > - score - new score

```	
CRAM.load(fname)
```
This function loads a trained model from a file.

> ####Parameters:   

> > - fname - file name
