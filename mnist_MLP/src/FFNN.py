import gzip, cPickle, os
import numpy as np

def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(os.path.split(__file__)[0], "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    #print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()     
    
    return train_set[0], train_set[1], valid_set[0], valid_set[1], test_set[0], test_set[1]
    # train_set[0]: 50000 * 784, train_set[1]: 50000
    # valid_set[0]: 10000 * 784, valid_set[1]: 10000
    # test_set[0]:  10000 * 784, test_set[1]:  10000
    


##############################
# read data

# Xtrain, ytrain, XCV, yCV, Xtest, ytest = load_data("mnist.pkl.gz")

# Check the shape of these variables
# Hopefully numpy is helpful
# Xtrain = np.array(Xtrain)
# XCV    = np.array(XCV)
# Xtest  = np.array(Xtest)

# Check the sizes of these numpy arrays
# print Xtrain.shape

##############################
# initialize paramters

# theta <- some small values

# Wait a minute. Where are you gonna save your parameters?
# You can either (1) assign a vector variable, in such small networks,
# for each parameter you need, e.g., W1, W2, b_hid, b_out, or 
# (2) assign a large parameter vector, indexed by some indices.
 
##############################
# LEARNING !

#for epoch = 1..infinity:

    # for each data sample:
        # If we implement gradient descent in a full-batch manner,
        # ignore this loop. But we can still try out mini-batch, or online SGD

        ##############################
        # forward propagation (4 lines, 2 for each layer)

        # z_hid = ??
        # y_hid = ??

        # z_out = ??
        # y_out = ??

        # A numerical issue of softmax:
        # subtract the largest value of z_out to prevent overflow

        #############################
        # add cost function J (your belief), and
        # compute the partial derivative 

        # dJ_dy_out = ?
        # For softmax, we obtain dJ_dz_out directly, rather than dJ_dy_out.

        #############################
        # backpropagation (6 lines, 3 for each layer)

        # dJ_dz_out = ?

        # dJ_dy_hid = ?
        # dJ_dz_hid = ?

        # dJ_db_out = ?
        # dJ_db_hid = ?
        # dJ_dW2 = ?
        # dJ_dW1 = ?

        # update parameters
        # Theta <- Theta - alpha * Grad
 
    # After one or several epochs, we want to monitor the performance of our network,
    
    # predict(X_train)
    # predict(X_CV)
    
    # prediction is pretty the same as forward propagation.
    # Now, you may have realized we'd better write forwardprop as a function

#############################################
# finally, we report the test accuracy.
# no data snooping is allowed, which is actually inevitable in researches

# precit(X_test)