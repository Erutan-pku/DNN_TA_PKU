from FFNN import load_data
import numpy as np
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import RMSprop
from keras.regularizers import l2

global mnist_MLP_Parameters
mnist_MLP_Parameters = {
    'model_layers':{'denseSize':[512,512],'activation':'relu','dropout':0.2,'W_regularizer':None},
    'loss':'categorical_crossentropy', 
    'optimizer':RMSprop(),
    'batch_size':128,
    'nb_epoch':1,
}
# https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

def getModel() :
    Input_tensor = Input(shape=(784, ), dtype='float32', name='Input_tensor')
    dense_1 = Dense(512, activation='relu')(Input_tensor)
    dense_1 = Dropout(0.2)(dense_1)
    dense_2 = Dense(512, activation='relu')(dense_1)
    dense_2 = Dropout(0.2)(dense_2)
    output = Dense(10, activation='softmax')(dense_2)

    model = Model(input=Input_tensor, output=output)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model

def getModelPara(Para) :
    model_layers = Para['model_layers']
    layers_num = len(model_layers['denseSize'])
    activations = model_layers['activation']
    dropouts = model_layers['dropout']
    if not type(activations) is list :
        activations = [model_layers['activation'] for i in range(layers_num)]
    if not type(dropouts) is list :
        dropouts = [model_layers['dropout'] for i in range(layers_num)]
    assert len(activations) == layers_num
    assert len(dropouts)    == layers_num

    Input_tensor = Input(shape=(784, ), dtype='float32', name='Input_tensor')
    dense_compute = Input_tensor
    for i in range(layers_num) :
        #print model_layers['denseSize'][i]
        dense_compute = Dense(model_layers['denseSize'][i], activation=activations[i], W_regularizer=model_layers['W_regularizer'])(dense_compute)
        if not dropouts[i] == 0:
            dense_compute = Dropout(dropouts[i])(dense_compute)
        #print dense_compute._shape
    output = Dense(10, activation='softmax')(dense_compute)

    model = Model(input=Input_tensor, output=output)
    model.compile(loss=Para['loss'], optimizer=Para['optimizer'], metrics=['accuracy'])
    return model


def minist_MLP(Para=None):
    if Para is None :
        Para = mnist_MLP_Parameters

    Xtrain, ytrain, XCV, yCV, Xtest, ytest = load_data("../mnist.pkl.gz")
    Ytrain = np_utils.to_categorical(ytrain, 10)
    YCV    = np_utils.to_categorical(yCV   , 10)
    Ytest  = np_utils.to_categorical(ytest , 10)
    #print np.shape(Xtrain), np.shape(Ytrain)
    #print np.shape(XCV)   , np.shape(YCV)
    #print np.shape(Xtest) , np.shape(Ytest)
    
    MLP_Model = getModelPara(Para)
    histories, scores = [], []
    for i in range(Para['nb_epoch']) :
        history = MLP_Model.fit(Xtrain, Ytrain, batch_size=Para['batch_size'], nb_epoch=1, verbose=1, validation_data=(XCV, YCV))
        score = MLP_Model.evaluate(Xtest, Ytest, verbose=0)
        
        histories.append(history.history)
        scores.append({'test_acc':score[1], 'test_loss':score[0]})
    return {'history':histories, 'test_result':scores}

if __name__ == '__main__':
    print minist_MLP()

