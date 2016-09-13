import sys
import numpy as np
import lasagne
import cv2
from cram import CRAM
from sklearn.metrics import accuracy_score, precision_score, recall_score
from collections import Counter
from scipy import stats
import time
import openslide

SIZE = 10000

def main():

    # Load dataset
    npzfile = np.load('/home/nnauata/utils/dataset_slides_path.npz')
    X_train_path = npzfile['X_train']
    y_train = npzfile['y_train']
    X_val_path = npzfile['X_valid']
    y_val = npzfile['y_valid']

    # Process train and valid data
    y_train = y_train.astype('int32')
    y_val = y_val.astype('int32')
    
    # Define model
    n_batch = 1
    n_channels = 1
    img_height = SIZE
    img_width = SIZE

    input_shape = (n_batch, n_channels, img_height, img_width)
    print "input shape: " + str(X_train.shape)
    specs = {
        'input_shape':input_shape,
        'patch':64,
        'n_steps':8,
        'n_h_g':64,
        'n_h_l':64,
        'n_f_g':64,
        'n_f_h':64,
        'n_h_fc_1':256,
        'learning_rate':0.0001,
        'n_classes':2,
        'sigma':0.1,
        'patience':50,
        'filter_shape_1':(128, 1, 3, 3),
        'filter_shape_2':(128, 128, 3, 3),
        'stride':(1, 1),
        'pooling_shape':(2, 2),
        'n_trials':2
        }

    # Define CRAM
    start = time.time()
    cram = CRAM(specs)
    print("Compilation time: " + str(time.time()-start))

    print "Start Preprocessing ..."
    X_train_obj = [openslide.OpenSlide(path) for path in X_train_path]
    X_val_obj = [openslide.OpenSlide(path) for path in X_val_path]

    print "Start Training ..."
    train_with_sgd(cram, input_shape, X_train_obj, y_train, X_val_obj, y_val, callback=sgd_callback)

    print "Start Preprocessing ..."
    X_test = npzfile['X_test']
    y_test = npzfile['y_test']
    X_test_obj = [openslide.OpenSlide(path) for path in X_train_path]

    print "Start Testing ..."
    # Predict a sample image
    cram.load('trained_model.npz')
    outs = []
    for x, y in zip(X_test, y_test):
        # Show some results
        X_in = np.reshape(x, input_shape)
        outs.append(cram.predict(X_in))
        # Show some results
        #l_t = cram.propose_region(X_in)
        #l_t = np.clip(np.reshape(l_t, (8, 2)), -1, 1)
        #print l_t
        #for k, l in enumerate(l_t):
        #    img = rho(l, x)
        #    cv2.imshow('patch_'+str(k), img.astype('uint8'))
        #cv2.imshow('img', x.astype('uint8'))
        #cv2.waitKey(0)

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

def train_with_sgd(model, input_shape, X_train_obj, y_train, X_valid_obj, y_valid, nepoch=10000000, callback_every=200, callback=None):

    # Init Early Stopping
    num_examples_seen = 0
    valid_loss = model.calculate_total_loss(X_valid_obj, y_valid)
    model.init_score(valid_loss)

    # Run epochs
    for epoch in range(nepoch):
        # For each training example...
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            X_in = np.reshape(read_image_openslide(X_train[i]), input_shape)
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

def read_image_openslide(full_path):
    openslide.OpenSlide(full_path)
    s = openslide.OpenSlide(full_path)
    k = slide.level_count - 1
    dim = slide.level_dimensions[k]
    img = np.array(slide.read_region((0,0), k, dim)) 
    final_img = Image.fromarray(img).convert("L")
    return final_img.resize((SIZE, SIZE))

if __name__ == "__main__":
    main()

