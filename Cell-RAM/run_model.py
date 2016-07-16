import sys
import numpy as np
import lasagne
import cv2
from cram import CRAM
from sklearn.metrics import accuracy_score, precision_score, recall_score
from collections import Counter
from scipy import stats
import time

def main():

    # Load dataset
    npzfile = np.load('../LSTM/dataset_large.npz')
    X_train = npzfile['X_train']
    y_train = npzfile['y_train']
    X_val = npzfile['X_valid']
    y_val = npzfile['y_valid']
    X_test = npzfile['X_test']
    y_test = npzfile['y_test']

    n_channels = 1
    img_height = X_train.shape[1]
    img_width = X_train.shape[2]

    # Preprocess dataset
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')

    X_train_original = X_train
    X_val_original = X_val
    X_test_original = X_test

    y_train = y_train.astype('int32')
    y_val = y_val.astype('int32')
    y_test = y_test.astype('int32')

    X_train = stats.zscore(X_train, axis=0)
    X_val = stats.zscore(X_val, axis=0)
    X_test = stats.zscore(X_test, axis=0)

    # Step1b: init variables
    n_batch = 1

    # Define model
    input_shape = (n_batch, n_channels, img_height, img_width)
    specs = {
        'input_shape':input_shape,
        'patch':64,
        'n_steps':8,
        'n_h_g':64,
        'n_h_l':64,
        'n_f_g':64,
        'n_f_h':64,
        'n_h_fc_1':256,
        'learning_rate':0.00001,
        'n_classes':2,
        'sigma':0.1,
        'patience':20,
        'filter_shape_1':(128, 1, 3, 3),
        'filter_shape_2':(128, 128, 3, 3),
        'stride':(1, 1),
        'pooling_shape':(2, 2),
        'n_trials':4
        }

    # Define CRAM
    start = time.time()
    cram = CRAM(specs)
    print("Compilation time: " + str(time.time()-start))

    #X_in = np.reshape(X_train[0], input_shape)
    #print cram.debug(X_in, [y_train[0]])[0]

    print "Start Training ..."
    train_with_sgd(cram, input_shape, X_train, y_train, X_val, y_val, callback=sgd_callback)

    print "Start Testing"
    # Predict a sample image
    cram.load('trained_model.npz')
    outs = []
    for x, y in zip(X_test, y_test):
        # Show some results
        X_in = np.reshape(x, input_shape)
        outs.append(cram.predict(X_in))
        # Show some results
        l_t = cram.propose_region(X_in)
        l_t = np.clip(np.reshape(l_t, (8, 2)), -1, 1)
        print l_t
        for k, l in enumerate(l_t):
            img = rho(l, x)
            cv2.imshow('patch_'+str(k), img.astype('uint8'))
        cv2.imshow('img', x.astype('uint8'))
        cv2.waitKey(0)

    print np.array(outs).ravel().shape
    outs = np.array(outs).ravel()
    print outs
    print y_test
    print(Counter(outs))
    print(Counter(y_test))

    print "Precision:"
    print(precision_score(y_test, outs))

    print "Recall"
    print(recall_score(y_test, outs))

    print "Accuracy:"
    print(accuracy_score(y_test, outs))

    print "Loss:"
    print cram.calculate_total_loss(X_test, y_test)
    return

def train_with_sgd(model, input_shape, X_train, y_train, X_valid, y_valid, nepoch=10000000, callback_every=200, callback=None):


    # Init Early Stopping
    num_examples_seen = 0
    valid_loss = model.calculate_total_loss(X_valid, y_valid)
    model.init_score(valid_loss)

    # Run epochs
    for epoch in range(nepoch):
        # For each training example...
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            X_in = np.reshape(X_train[i], input_shape)
            #print model.debug(X_in, [y_train[i]])[0]
            model.fit(X_in, [y_train[i]])

            num_examples_seen += 1
            # Optionally do callback
            if (callback and callback_every and num_examples_seen % callback_every == 0):
                callback(model, num_examples_seen, X_train, y_train, X_valid, y_valid)

            # Check patience
            if model.is_finished(): return
    return

def sgd_callback(model, num_examples_seen, X_train, y_train, X_valid, y_valid):

    print "Iteration:"
    print("(%d)" % (num_examples_seen))
    print("--------------------------------------------------")
    # Compute loss
    valid_loss = model.calculate_total_loss(X_valid, y_valid)
    reinforce_loss = model.calculate_proposal_loss(X_valid, y_valid)

    print("Valid Loss")
    print("Best So Far: %f" % model.get_score())
    print("Next Candidate: %f" % valid_loss)
    print("Reinforce Loss: %f -- included in the validation" %reinforce_loss)

    # Perform Early Stopping
    model.update_score(valid_loss)

    # Show some results
    #l_t = model.propose_region(np.reshape(X_train[0], (1, 1, 500, 500)))
    #l_t = np.clip(np.reshape(l_t, (2, 2)), -1, 1)
    #print l_t
    #for k, l in enumerate(l_t):
    #    img = rho(l, X_train[0])
    #    cv2.imshow('patch_'+str(k), img.astype('uint8'))
    #cv2.imshow('img', X_train[0].astype('uint8'))
    #cv2.waitKey(0)
    return

def rho(l_tp1, x):
    PATCH_SIZE = 64
    y_start = int((1 + l_tp1[0]) * (500 - PATCH_SIZE) / 2)
    x_start = int((1 + l_tp1[1]) * (500 - PATCH_SIZE) / 2)
    img = x[y_start:y_start+PATCH_SIZE, x_start:x_start+PATCH_SIZE]
    return img
if __name__ == "__main__":
    main()

