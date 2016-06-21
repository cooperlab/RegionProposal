import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator
import cv2
from layers import GRULayer, InnerProductLayer
from models import RecurrentNeuralNetwork, MultilayerPerceptron
import sys
from scipy import stats
import numpy
from sklearn.metrics import accuracy_score
numpy.set_printoptions(threshold=numpy.nan)
from sklearn import svm
from sklearn.decomposition import PCA

def main():

    # FAZER LAYER PARA HISTOGRAM!!!!
    # UNIFICAR AS DUAS LISTAS DE LAYERS!!!
    # IMPLEMENTAR FUNCAO HISTOGRAM EM THEANO!!!
    layers = [GRULayer(hidden_dim=256), GRULayer(hidden_dim=256), GRULayer(hidden_dim=256)]
    pred_layers = [InnerProductLayer(in_dim=65536, out_dim=256), InnerProductLayer(in_dim=256, out_dim=1)]
    rnn = RecurrentNeuralNetwork(layers, pred_layers)
    #mlp = MultilayerPerceptron(pred_layers)

    npzfile = np.load('./dataset_small.npz')
    X_train = npzfile['X_train']
    y_train = npzfile['y_train']
    X_valid = npzfile['X_valid']
    y_valid = npzfile['y_valid']
    X_test = npzfile['X_test']
    y_test = npzfile['y_test']

    X_train = stats.zscore(X_train, axis=0)
    X_valid = stats.zscore(X_valid, axis=0)

    #X_train = np.array([X_train[i].ravel() for i in range(len(y_train))])
    #X_valid = np.array([X_valid[i].ravel() for i in range(len(y_valid))])
    #X_test = np.array([X_test[i].ravel() for i in range(len(y_test))])

    #pca = PCA(n_components=65536)
    #pca.fit(X_train)
    #X_train = pca.transform(X_train)
    #X_valid = pca.transform(X_valid)
    #X_test = pca.transform(X_test)

    print('Input size:')
    print("--------------------------------------------------")
    print(X_train.shape)
    print(y_train.shape)

    # SVM EXP
    #clf = svm.SVC()
    #clf.fit(X_train, y_train)
    #preds = clf.predict(X_test)
    #print(accuracy_score(y_test, preds))

    train_with_sgd(rnn, X_train, y_train, X_valid, y_valid, learning_rate=0.00001, nepoch=100000, decay=0.98,
                   callback=sgd_callback)

    # Predict a sample image
    rnn.load('trained_model.npz')
    outs = []
    for x, y in zip(X_train, y_train):

        # Show some results
        outs.append(rnn.predict(x))

        cv2.imshow('img', x)
        m = rnn.get_mask(x)
        cv2.imshow('mask', m)
        p = rnn.propose_region(x)
        cv2.imshow('pred', p)
        #out_filtered = np.where(out >= 0.5, 1.0, 0.0)
        #cv2.imshow('pred_filtered', out_filtered)
        cv2.waitKey(0)
    outs = np.where(np.array(outs).ravel() >= 0.5, 1, 0)
    print y_test, outs
    print(accuracy_score(y_test,outs))
    return

def train_with_sgd(model, X_train, y_train, X_valid, y_valid, learning_rate=0.01, nepoch=20, decay=0.9,
    callback_every=200, callback=None):
    num_examples_seen = 0

    # Init Early Stopping
    # Show some results
    valid_loss = model.calculate_total_loss(X_valid, y_valid)
    model.best_score = valid_loss

    # Run epochs
    for epoch in range(nepoch):
        # For each training example...
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], [[y_train[i]]], learning_rate, decay)
            num_examples_seen += 1
            # Optionally do callback
            if (callback and callback_every and num_examples_seen % callback_every == 0):
                callback(model, num_examples_seen, X_train, y_train, X_valid, y_valid)

        # Check patience
        if model.patience == 0:
            model.load('trained_model.npz')
            return
    return

def sgd_callback(model, num_examples_seen, X_train, y_train, X_valid, y_valid):

    print "Iteration:"
    print("(%d)" % (num_examples_seen))
    print("--------------------------------------------------")
    # Compute loss
    #train_loss = model.calculate_total_loss(X_train, y_train)
    valid_loss = model.calculate_total_loss(X_valid, y_valid)

    print("Valid Loss")
    print("Best So Far: %f" % model.best_score)
    print("Next Candidate: %f" % valid_loss)
    # print("Train Loss: %f" % train_loss)

    # Perform Early Stopping
    if valid_loss < model.best_score:

        print('Saving model to trained_model.npz')
        model.save('trained_model.npz')
        model.reset_patience()
        model.best_score = valid_loss

    else:
        model.decrease_patience()
        print('Patience:')
        print(model.patience)

    print("\n")
    sys.stdout.flush()

    # Show some results
    out = model.propose_region(X_train[0])
    mask = model.get_mask(X_train[0])
    cv2.imshow('mask', mask)
    cv2.imshow('img', X_train[0])
    cv2.imshow('pred', out)
    cv2.waitKey(0)
    return

if __name__ == "__main__":
    main()