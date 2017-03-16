from FFNN import load_data
import numpy as np
import codecs
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import RMSprop
from keras.regularizers import activity_l1
from mnist_MLP_tune import resultAnalisys
from keras.optimizers import *

global output
output = codecs.open('mnist_MLP_AutoEncoder.result', "w", "utf-8")

def main(denses=[32], tttt=1):
    verbose = 1
    #Data
    Xtrain, ytrain, XCV, yCV, Xtest, ytest = load_data("../mnist.pkl.gz")
    Ytrain = np_utils.to_categorical(ytrain, 10)
    YCV    = np_utils.to_categorical(yCV   , 10)
    Ytest  = np_utils.to_categorical(ytest , 10)
    #print np.shape(Xtrain), np.shape(Ytrain)
    #print np.shape(XCV)   , np.shape(YCV)
    #print np.shape(Xtest) , np.shape(Ytest)
    x_last_train = Xtrain
    x_last_cv = XCV
    x_last_test = Xtest

    #parameters
    activity_regularizer=activity_l1(10e-5)

    # predicted model
    Input_tensor = Input(shape=(denses[-1], ), dtype='float32')
    tensor_compute = Dense(32, activation='relu')(Input_tensor)
    tensor_compute = Dense(32, activation='relu')(tensor_compute)
    output = Dense(10, activation='softmax')(tensor_compute)
    model = Model(input=Input_tensor, output=output)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    # AutoEncoder notFixed
    #"""
    if tttt == 1 :
        Input_tensor = Input(shape=(784, ), dtype='float32')
        encoder_tensor = Input_tensor
        for dense_size in denses:
            encoder_tensor = Dense(dense_size, activation='relu', activity_regularizer=activity_regularizer)(encoder_tensor)
        EncoderModel = Model(input=Input_tensor, output=encoder_tensor)
        decoder_tensor = encoder_tensor
        for dense_size in denses[::-1][1:] :
            decoder_tensor = Dense(dense_size, activation='relu')(decoder_tensor)
        decoder_tensor = Dense(784, activation='sigmoid')(decoder_tensor)
        AutoEncoderModel = Model(input=Input_tensor, output=decoder_tensor)

        AutoEncoderModel.compile(optimizer='adadelta', loss='binary_crossentropy')
        result_aem = AutoEncoderModel.fit(
                Xtrain, 
                Xtrain,
                nb_epoch=10,
                batch_size=256,
                shuffle=True,
                verbose=verbose,
                validation_data=(XCV, XCV))
        print result_aem.history['loss'][-1]
        x_last_train = EncoderModel.predict(x_last_train, batch_size=1)
        x_last_cv = EncoderModel.predict(x_last_cv, batch_size=1)
        x_last_test = EncoderModel.predict(x_last_test, batch_size=1)
        print np.shape(x_last_train)
        print np.shape(x_last_cv)
        print np.shape(x_last_test)

    #"""
    #
    #
    #
    #"""
    # AutoEncoder Fixed
    elif tttt == 2 :
        denseSize = [784] + denses
        for i in range(len(denseSize)-1) :
            #A AutoEncoderLayer
            Input_tensor = Input(shape=(denseSize[i], ), dtype='float32')
            encoded = Dense(denseSize[i+1], activation='relu', activity_regularizer=activity_regularizer)(Input_tensor)
            EncoderModel = Model(input=Input_tensor, output=encoded)
            decoded = Dense(denseSize[i], activation='sigmoid', activity_regularizer=activity_regularizer)(encoded)
            AutoEncoderModel = Model(input=Input_tensor, output=decoded)

            AutoEncoderModel.compile(optimizer='adadelta', loss='mean_absolute_error')
            result_aem = AutoEncoderModel.fit(
                x_last_train, 
                x_last_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                verbose=verbose,
                validation_data=(x_last_test, x_last_test))
            print result_aem.history['loss'][-1]

            x_last_train = EncoderModel.predict(x_last_train, batch_size=1)
            x_last_cv = EncoderModel.predict(x_last_cv, batch_size=1)
            x_last_test = EncoderModel.predict(x_last_test, batch_size=1)
            print np.shape(x_last_train)
            print np.shape(x_last_cv)
            print np.shape(x_last_test)
    else :
        denseSize = [784] + denses
        for i in range(len(denseSize)-1) :
            #A AutoEncoderLayer
            Input_tensor = Input(shape=(denseSize[i], ), dtype='float32')
            encoded = Dense(denseSize[i+1], activation='relu', activity_regularizer=activity_regularizer)(Input_tensor)
            EncoderModel = Model(input=Input_tensor, output=encoded)
            decoded = Dense(784, activation='sigmoid', activity_regularizer=activity_regularizer)(encoded)
            AutoEncoderModel = Model(input=Input_tensor, output=decoded)

            AutoEncoderModel.compile(optimizer='adadelta', loss='binary_crossentropy')
            result_aem = AutoEncoderModel.fit(
                x_last_train, 
                Xtrain,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                verbose=verbose,
                validation_data=(x_last_test, Xtest))
            print result_aem.history['loss'][-1]

            x_last_train = EncoderModel.predict(x_last_train, batch_size=1)
            x_last_cv = EncoderModel.predict(x_last_cv, batch_size=1)
            x_last_test = EncoderModel.predict(x_last_test, batch_size=1)
            print np.shape(x_last_train)
            print np.shape(x_last_cv)
            print np.shape(x_last_test)

    
    #"""

    # test
    histories, scores = [], []
    for i in range(100) :
        history = model.fit(x_last_train, Ytrain, batch_size=128, nb_epoch=1, verbose=verbose, validation_data=(x_last_cv, YCV))
        score = model.evaluate(x_last_test, Ytest, verbose=0)
        
        histories.append(history.history)
        scores.append({'test_acc':score[1], 'test_loss':score[0]})
    resultAnalisys({'history':histories, 'test_result':scores})
    print histories[-1]

if __name__ == '__main__':
    #main(denses=[128,64,32], tttt=3)
    #"""
    main(denses=[32], tttt=1)
    main(denses=[128,64,32], tttt=1)
    main(denses=[512,128], tttt=1)
    main(denses=[32], tttt=2)
    main(denses=[128,64,32], tttt=2)
    main(denses=[512,128], tttt=2)
    main(denses=[32], tttt=3)
    main(denses=[128,64,32], tttt=3)
    main(denses=[512,128], tttt=3)
    #"""
"""
0.0996965091848
(50000, 32)
(10000, 32)
(10000, 32)
86 0.9668
86 0.9661 0.9668
{'acc': [0.971619999961853], 'loss': [0.090356856570243838], 'val_acc': [0.9617], 'val_loss': [0.13214773284345865]}

0.107046216388
(50000, 32)
(10000, 32)
(10000, 32)
77 0.9658
95 0.9667 0.9639
{'acc': [0.96805999998092651], 'loss': [0.10176492255449295], 'val_acc': [0.96640000000000004], 'val_loss': [0.11867902052402496]}

0.0759504167557
(50000, 128)
(10000, 128)
(10000, 128)
97 0.9698
52 0.9684 0.9653
{'acc': [0.97713999996185308], 'loss': [0.073015511084794996], 'val_acc': [0.96360000000000001], 'val_loss': [0.14942134196162224]}

0.0905377820539
(50000, 32)
(10000, 32)
(10000, 32)
96 0.9389
83 0.9417 0.9373
{'acc': [0.93940000001907353], 'loss': [0.19284992527484893], 'val_acc': [0.94030000000000002], 'val_loss': [0.20032406220436097]}

0.0834107968664
(50000, 128)
(10000, 128)
(10000, 128)
2.31844632759
(50000, 64)
(10000, 64)
(10000, 64)
6.03984785446
(50000, 32)
(10000, 32)
(10000, 32)
96 0.9088
96 0.9071 0.9088
{'acc': [0.89727999998092656], 'loss': [0.32630057114601135], 'val_acc': [0.88580000000000003], 'val_loss': [0.36694348468780519]}

0.0729745532918
(50000, 512)
(10000, 512)
(10000, 512)
0.905748859367
(50000, 128)
(10000, 128)
(10000, 128)
84 0.9620
83 0.9631 0.9588
{'acc': [0.96548], 'loss': [0.11161914931774139], 'val_acc': [0.95840000000000003], 'val_loss': [0.15851176904216407]}

0.109844921024
(50000, 32)
(10000, 32)
(10000, 32)
74 0.9680
63 0.9665 0.9630
{'acc': [0.97284000001907345], 'loss': [0.086112538641691203], 'val_acc': [0.96099999999999997], 'val_loss': [0.14643984887343831]}

0.0916227527618
(50000, 128)
(10000, 128)
(10000, 128)
0.0911646131682
(50000, 64)
(10000, 64)
(10000, 64)
0.106085488768
(50000, 32)
(10000, 32)
(10000, 32)
88 0.9625
80 0.9655 0.9608
{'acc': [0.9687800000190735], 'loss': [0.10335537680864335], 'val_acc': [0.9577], 'val_loss': [0.15805749633312224]}

0.083424787457
(50000, 512)
(10000, 512)
(10000, 512)
0.0834579735875
(50000, 128)
(10000, 128)
(10000, 128)
67 0.9708
83 0.9713 0.9701
{'acc': [0.97322000001907349], 'loss': [0.087526985633373258], 'val_acc': [0.96540000000000004], 'val_loss': [0.14346503933519125]}
"""

