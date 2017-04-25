# coding=utf-8
# -*- coding: UTF-8 -*- 
import sys
from keras.models import Model
from keras.layers import Input, Dropout, Flatten, Dense, Lambda, Activation, add, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
import keras
import tensorflow as tf

Model_Feature = {
    'num_classes' : 17 , 
    'InputShape'  : (299, 299, 3, )
}

def getModel() :
    input_img = Input(shape=Model_Feature['InputShape'], dtype='float32', name = 'input_img')
    """ MLP
    t = Flatten()(input_img)
    t = Dense(128, activation='relu')(t)
    t = Dense(128, activation='relu')(t)
    t = Dense(128, activation='relu')(t)
    #"""
    """ CNN_simple
    t = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_img)
    print t._shape
    t = MaxPooling2D(pool_size=(2, 2))(t)
    print t._shape
    t = Conv2D(32, kernel_size=(3, 3), activation='relu')(t)
    print t._shape
    t = MaxPooling2D(pool_size=(2, 2))(t)
    print t._shape
    t = Flatten()(t)

    t = Dense(128, activation='relu')(t)
    t = Dense(128, activation='relu')(t)
    #"""
    """ CNN_Norm
    t = Conv2D(16, kernel_size=(3, 3), activation='relu')(input_img)
    print t._shape
    t = MaxPooling2D(pool_size=(2, 2))(t)
    #t = BatchNormalization()(t)
    t = Dropout(0.25)(t)
    print t._shape
    t = Conv2D(32, kernel_size=(3, 3), activation='relu')(t)
    print t._shape
    t = MaxPooling2D(pool_size=(2, 2))(t)
    #t = BatchNormalization()(t)
    t = Dropout(0.25)(t)
    print t._shape
    t = Conv2D(64, kernel_size=(3, 3), activation='relu')(t)
    print t._shape
    t = MaxPooling2D(pool_size=(2, 2))(t)
    #t = BatchNormalization()(t)
    t = Dropout(0.25)(t)
    print t._shape
    t = Flatten()(t)

    t = Dense(128, activation='relu')(t)
    #t = Dropout(0.5)(t)
    t = Dense(128, activation='relu')(t)
    #t = Dropout(0.5)(t)

    #"""
    t = Dense(Model_Feature['num_classes'], activation='softmax')(t)
    print t._shape
    

    model = Model(inputs=input_img, outputs=t)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

def getModel_AlexNet() :
    local_response_normalization = Lambda(lambda x : tf.nn.local_response_normalization(x, depth_radius=2, bias=2, alpha=1e-4, beta=0.75))
    # http://blog.csdn.net/mao_xiao_feng/article/details/53488271
    # https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization

    # lightest
    input_img = Input(shape=Model_Feature['InputShape'], dtype='float32', name = 'input_img')
    #227*227*3
    
    #t = Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu')(input_img)
    t = Conv2D(72, kernel_size=(11, 11), strides=(4, 4), activation='relu')(input_img)
    #t = Conv2D(96, kernel_size=(7, 7), strides=(4, 4), activation='relu')(input_img)
    #t = Conv2D(72, kernel_size=(7, 7), strides=(4, 4), activation='relu')(input_img)
    print t._shape
    # 55*55*96

    t = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(t)
    print t._shape

    t = local_response_normalization(t)
    #t = BatchNormalization()(t)
    print t._shape
    # 27*27*96

    #t = Conv2D(256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(t)
    #t = Conv2D(192, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(t)
    t = Conv2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(t)
    print t._shape
    # 27*27*96

    t = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(t)
    print t._shape
    # 13*13*96

    t = local_response_normalization(t)
    #t = BatchNormalization()(t)
    print t._shape

    #t = Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(t)
    #print t._shape
    # 13*13*384

    #t = Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(t)
    #t = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(t)
    #t = Conv2D(192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(t)
    t = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(t)
    print t._shape
    # 13*13*384

    #t = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(t)
    #t = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(t)
    #t = Conv2D(96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(t)
    t = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(t)
    print t._shape
    # 13*13*256

    t = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(t)
    print t._shape
    # 6*6*256

    t = Flatten()(t)
    #t = Dense(4096, activation='relu')(t)
    #t = Dense(1024, activation='relu')(t)
    #t = Dense(512, activation='relu')(t)
    t = Dense(128, activation='relu')(t)
    t = Dropout(0.5)(t)
    print t._shape
    # 4096

    #t = Dense(4096, activation='relu')(t)
    #t = Dense(1024, activation='relu')(t)
    #t = Dense(512, activation='relu')(t)
    t = Dense(128, activation='relu')(t)
    t = Dropout(0.5)(t)
    print t._shape
    # 4096

    t = Dense(Model_Feature['num_classes'], activation='softmax')(t)
    print t._shape

    model = Model(inputs=input_img, outputs=t)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

def getModel_VGG13():
    #参考 https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py
    input_img = Input(shape=Model_Feature['InputShape'], dtype='float32', name = 'input_img')
    #227*227*3

    # Block 1
    t = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
    t = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(t)
    t = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(t)
    print t._shape
    # 113*113*64

    # Block 2
    t = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(t)
    t = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(t)
    t = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(t)
    print t._shape
    # 56*56*128

    # Block 3
    t = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(t)
    t = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(t)
    t = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(t)
    print t._shape
    # 28*28*256

    # Block 4
    t = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(t)
    t = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(t)
    t = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(t)
    print t._shape
    # 14*14*512

    # Block 5
    t = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(t)
    t = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(t)
    t = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(t)
    print t._shape
    # 7*7*512

    # Classification block
    t = Flatten()(t)
    t = Dense(4096, activation='relu')(t)
    t = Dense(4096, activation='relu')(t)
    print t._shape
    # 4096
    t = Dense(Model_Feature['num_classes'], activation='softmax')(t)
    print t._shape
    # 17

    model = Model(inputs=input_img, outputs=t)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

def identity_block(input_tensor, filters):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, kernel_size=(1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #print x._shape

    x = Conv2D(filters2, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #print x._shape

    x = Conv2D(filters3, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    #print x._shape

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    #print x._shape
    return x
def conv_block(input_tensor, filters, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, kernel_size=(1, 1), strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #print x._shape

    x = Conv2D(filters2, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #print x._shape

    x = Conv2D(filters3, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    #print x._shape

    shortcut = Conv2D(filters3, kernel_size=(1, 1), strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    #print x._shape

    x = add([x, shortcut])
    x = Activation('relu')(x)
    #print x._shape
    return x
def identity_block_2(input_tensor, filters):
    filters1, filters2 = filters

    x = Conv2D(filters1, kernel_size=(3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #print x._shape

    x = Conv2D(filters2, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    #print x._shape

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    #print x._shape
    return x
def conv_block_2(input_tensor, filters, strides=(2, 2)):
    filters1, filters2 = filters

    x = Conv2D(filters1, kernel_size=(3, 3), strides=strides, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #print x._shape

    x = Conv2D(filters2, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    #print x._shape

    shortcut = Conv2D(filters2, kernel_size=(1, 1), strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    #print shortcut._shape

    x = add([x, shortcut])
    x = Activation('relu')(x)
    #print x._shape
    return x
def getResNet101() :
    # 参考 https://zhuanlan.zhihu.com/p/21586417?refer=chicken-life
    # 参考 https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py
    input_img = Input(shape=Model_Feature['InputShape'], dtype='float32', name = 'input_img')
    #227*227*3


    # x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(input_img)
    print x._shape
    # (?, 111, 111, 64)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    print x._shape
    # (?, 55, 55, 64)

    x = conv_block(x, filters=[64, 64, 256], strides=(1, 1))
    x = identity_block(x, filters=[64, 64, 256])
    x = identity_block(x, filters=[64, 64, 256])
    print x._shape

    x = conv_block(x, filters=[128, 128, 512])
    x = identity_block(x, filters=[128, 128, 512])
    x = identity_block(x, filters=[128, 128, 512])
    x = identity_block(x, filters=[128, 128, 512])
    print x._shape

    x = conv_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    print x._shape

    x = conv_block(x, filters=[512, 512, 2048])
    x = identity_block(x, filters=[512, 512, 2048])
    x = identity_block(x, filters=[512, 512, 2048])
    print x._shape
    # (?, 7, 7, 2048)

    x = AveragePooling2D(pool_size=(7, 7))(x)
    print x._shape
    # (?, 1, 1, 2048)

    x = Flatten()(x)
    x = Dense(Model_Feature['num_classes'], activation='softmax')(x)
    print x._shape
    # 17

    model = Model(inputs=input_img, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model
def getResNet50() :
    # 参考 https://zhuanlan.zhihu.com/p/21586417?refer=chicken-life
    # 参考 https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py
    input_img = Input(shape=Model_Feature['InputShape'], dtype='float32', name = 'input_img')
    #227*227*3


    # x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(input_img)
    print x._shape
    # (?, 111, 111, 64)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    print x._shape
    # (?, 55, 55, 64)

    x = conv_block(x, filters=[64, 64, 256], strides=(1, 1))
    x = identity_block(x, filters=[64, 64, 256])
    x = identity_block(x, filters=[64, 64, 256])
    print x._shape

    x = conv_block(x, filters=[128, 128, 512])
    x = identity_block(x, filters=[128, 128, 512])
    x = identity_block(x, filters=[128, 128, 512])
    x = identity_block(x, filters=[128, 128, 512])
    print x._shape

    x = conv_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    x = identity_block(x, filters=[256, 256, 1024])
    print x._shape

    x = conv_block(x, filters=[512, 512, 2048])
    x = identity_block(x, filters=[512, 512, 2048])
    x = identity_block(x, filters=[512, 512, 2048])
    print x._shape
    # (?, 7, 7, 2048)

    x = AveragePooling2D(pool_size=(7, 7))(x)
    print x._shape
    # (?, 1, 1, 2048)

    x = Flatten()(x)
    x = Dense(Model_Feature['num_classes'], activation='softmax')(x)
    print x._shape
    # 17

    model = Model(inputs=input_img, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model
def getResNet34() :
    # 参考 https://zhuanlan.zhihu.com/p/21586417?refer=chicken-life
    # 参考 https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py
    input_img = Input(shape=Model_Feature['InputShape'], dtype='float32', name = 'input_img')
    #227*227*3


    # x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(input_img)
    print x._shape
    # (?, 111, 111, 64)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    print x._shape
    # (?, 55, 55, 64)

    x = conv_block_2(x, filters=[64, 64], strides=(1, 1))
    x = identity_block_2(x, filters=[64, 64])
    x = identity_block_2(x, filters=[64, 64])
    print x._shape

    x = conv_block_2(x, filters=[128, 128])
    x = identity_block_2(x, filters=[128, 128])
    x = identity_block_2(x, filters=[128, 128])
    x = identity_block_2(x, filters=[128, 128])
    print x._shape

    x = conv_block_2(x, filters=[256, 256])
    x = identity_block_2(x, filters=[256, 256])
    x = identity_block_2(x, filters=[256, 256])
    x = identity_block_2(x, filters=[256, 256])
    x = identity_block_2(x, filters=[256, 256])
    x = identity_block_2(x, filters=[256, 256])
    print x._shape

    x = conv_block_2(x, filters=[512, 512])
    x = identity_block_2(x, filters=[512, 512])
    x = identity_block_2(x, filters=[512, 512])
    print x._shape
    # (?, 7, 7, 2048)

    x = AveragePooling2D(pool_size=(7, 7))(x)
    print x._shape
    # (?, 1, 1, 2048)

    x = Flatten()(x)
    x = Dense(Model_Feature['num_classes'], activation='softmax')(x)
    print x._shape
    # 17

    model = Model(inputs=input_img, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model
def getResNet18() :
    # 参考 https://zhuanlan.zhihu.com/p/21586417?refer=chicken-life
    # 参考 https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py
    input_img = Input(shape=Model_Feature['InputShape'], dtype='float32', name = 'input_img')
    #227*227*3


    # x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(input_img)
    print x._shape
    # (?, 111, 111, 64)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    print x._shape
    # (?, 55, 55, 64)

    x = conv_block_2(x, filters=[64, 64], strides=(1, 1))
    x = identity_block_2(x, filters=[64, 64])
    print x._shape

    x = conv_block_2(x, filters=[128, 128])
    x = identity_block_2(x, filters=[128, 128])
    print x._shape

    x = conv_block_2(x, filters=[256, 256])
    x = identity_block_2(x, filters=[256, 256])
    print x._shape

    x = conv_block_2(x, filters=[512, 512])
    x = identity_block_2(x, filters=[512, 512])
    print x._shape
    # (?, 7, 7, 2048)

    x = AveragePooling2D(pool_size=(7, 7))(x)
    print x._shape
    # (?, 1, 1, 2048)

    x = Flatten()(x)
    x = Dense(Model_Feature['num_classes'], activation='softmax')(x)
    print x._shape
    # 17

    model = Model(inputs=input_img, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

def inception(input_tensor, paras) :
    tensor_a = Conv2D(paras[0], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    #print tensor_a._shape

    tensor_b = Conv2D(paras[1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    tensor_b = Conv2D(paras[2], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(tensor_b)
    #print tensor_b._shape

    tensor_c = Conv2D(paras[3], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    tensor_c = Conv2D(paras[4], kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(tensor_c)
    #print tensor_c._shape

    tensor_d = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    tensor_d = Conv2D(paras[5], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(tensor_d)
    #print tensor_d._shape

    output_tensor = concatenate([tensor_a,tensor_b,tensor_c,tensor_d])
    #print output_tensor._shape
    return output_tensor
def getGooleNetOut(input_tensor) :
    t = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(input_tensor)
    #print t._shape

    t = Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same')(t)
    #print t._shape

    t = Flatten()(t)
    t = Dense(1024, activation='relu')(t)
    t = Dropout(0.7)(t)
    #print t._shape

    t = Dense(Model_Feature['num_classes'], activation='softmax')(t)
    #print t._shape
    return t
def getGoogleNet_v1() :
    # 参考 http://blog.csdn.net/wang4959520/article/details/51832233  
    local_response_normalization = Lambda(lambda x : tf.nn.local_response_normalization(x, depth_radius=2, bias=2, alpha=1e-4, beta=0.75))
    # http://blog.csdn.net/mao_xiao_feng/article/details/53488271
    # https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization

    input_img = Input(shape=Model_Feature['InputShape'], dtype='float32', name = 'input_img')
    #227*227*3

    t = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), activation='relu')(input_img)
    print t._shape
    # (?, 114, 114, 64)

    t = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(t)
    t = local_response_normalization(t)
    print t._shape
    # (?, 56, 56, 64)

    #t = Conv2D(192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(t)
    #print t1._shape
    # (?, 56, 56, 192)

    t = Conv2D(192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(t)
    print t._shape
    # (?, 56, 56, 192)

    t = local_response_normalization(t)
    t = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(t)
    print t._shape
    # (?, 28, 28, 192)

    t = inception(t, paras=[64,96,128,16,32,32])
    print t._shape
    # (?, 28, 28, 256)
    
    t = inception(t, paras=[128,128,192,32,96,64])
    print t._shape
    # (?, 28, 28, 480)

    t = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(t)
    print t._shape
    # (?, 14, 14, 480)

    t = inception(t, paras=[192,96,208,16,48,64])
    print t._shape
    # (?, 14, 14, 512)

    result_1 = getGooleNetOut(t)
    print result_1._shape
    # (?, 17)

    t = inception(t, paras=[160,112,224,24,64,64])
    print t._shape
    # (?, 14, 14, 512)

    t = inception(t, paras=[128,128,256,24,64,64])
    print t._shape
    # (?, 14, 14, 512)

    t = inception(t, paras=[112,144,288,32,64,64])
    print t._shape
    # (?, 14, 14, 528)

    result_2 = getGooleNetOut(t)
    print result_2._shape

    t = inception(t, paras=[256,160,320,32,128,128])
    print t._shape
    # (?, 14, 14, 832)
    
    t = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(t)
    print t._shape
    # (?, 7, 7, 832)

    t = inception(t, paras=[256,160,320,32,128,128])
    print t._shape
    # (?, 7, 7, 832)

    t = inception(t, paras=[384,192,384,48,128,128])
    print t._shape
    # (?, 7, 7, 1024)

    t = AveragePooling2D(pool_size=(7, 7))(t)    
    print t._shape
    # (?, 1, 1, 1024)

    t = Flatten()(t)
    t = Dropout(0.4)(t)
    t = Dense(1000, activation='relu')(t)
    print t._shape
    # (?, 1000)

    t = Dense(Model_Feature['num_classes'], activation='softmax')(t)
    print t._shape
    # 17

    average_3 = Lambda(lambda x : (x[0] + x[1] + x[2]) / 3)
    t = average_3([t, result_2, result_1])
    print t._shape

    model = Model(inputs=input_img, outputs=t)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])  # , loss_weights=[.5, .3, .2]
    return model

def inceptionv2_1(input_tensor, paras) :
    tensor_a = Conv2D(paras[0], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    #print tensor_a._shape

    tensor_b = Conv2D(paras[1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    tensor_b = Conv2D(paras[2], kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(tensor_b)
    #print tensor_b._shape

    tensor_c = Conv2D(paras[3], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    tensor_c = Conv2D(paras[4], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(tensor_c)
    tensor_c = Conv2D(paras[5], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(tensor_c)
    #print tensor_c._shape

    tensor_d = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    tensor_d = Conv2D(paras[6], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(tensor_d)
    #print tensor_d._shape

    output_tensor = concatenate([tensor_a,tensor_b,tensor_c,tensor_d])
    #print output_tensor._shape
    return output_tensor
def inception_pad1(input_tensor, paras) :
    tensor_a = Conv2D(paras[0], kernel_size=(3, 3), strides=(2, 2), activation='relu')(input_tensor)
    #print tensor_a._shape

    tensor_b = Conv2D(paras[1], kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
    tensor_b = Conv2D(paras[2], kernel_size=(3, 3), activation='relu', padding='same')(tensor_b)
    tensor_b = Conv2D(paras[2], kernel_size=(3, 3), strides=(2, 2), activation='relu')(tensor_b)
    #print tensor_b._shape

    tensor_c = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(input_tensor)
    #print tensor_c._shape

    output_tensor = concatenate([tensor_a,tensor_b,tensor_c])
    #print output_tensor._shape
    return output_tensor
def inceptionv2_2(input_tensor, paras) :
    tensor_a = Conv2D(paras[0], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    #print tensor_a._shape

    tensor_b = Conv2D(paras[1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    tensor_b = Conv2D(paras[2], kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(tensor_b)
    tensor_b = Conv2D(paras[3], kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(tensor_b)
    #print tensor_b._shape

    tensor_c = Conv2D(paras[4], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    tensor_c = Conv2D(paras[5], kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(tensor_c)
    tensor_c = Conv2D(paras[6], kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(tensor_c)
    tensor_c = Conv2D(paras[7], kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(tensor_c)
    tensor_c = Conv2D(paras[8], kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(tensor_c)
    #print tensor_c._shape

    tensor_d = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    tensor_d = Conv2D(paras[9], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(tensor_d)
    #print tensor_d._shape

    output_tensor = concatenate([tensor_a,tensor_b,tensor_c,tensor_d])
    #print output_tensor._shape
    return output_tensor
def getGooleNetOut_v2(input_tensor) :
    t = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(input_tensor)
    #print t._shape

    t = Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same')(t)
    #print t._shape

    t = Flatten()(t)
    t = Dense(768, activation='relu')(t)
    #print t._shape

    t = Dense(Model_Feature['num_classes'], activation='softmax')(t)
    #print t._shape
    return t
def inception_pad2(input_tensor, paras) :
    tensor_a = Conv2D(paras[0], kernel_size=(1, 1), activation='relu')(input_tensor)
    tensor_a = Conv2D(paras[1], kernel_size=(3, 3), strides=(2, 2), activation='relu')(tensor_a)
    #print tensor_a._shape

    tensor_b = Conv2D(paras[2], kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
    tensor_b = Conv2D(paras[3], kernel_size=(1, 7), activation='relu', padding='same')(tensor_b)
    tensor_b = Conv2D(paras[4], kernel_size=(7, 1), activation='relu', padding='same')(tensor_b)
    tensor_b = Conv2D(paras[5], kernel_size=(3, 3), strides=(2, 2), activation='relu')(tensor_b)
    #print tensor_b._shape

    tensor_c = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(input_tensor)
    #print tensor_c._shape

    output_tensor = concatenate([tensor_a,tensor_b,tensor_c])
    #print output_tensor._shape
    return output_tensor
def inceptionv2_3(input_tensor, paras) :
    tensor_a = Conv2D(paras[0], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    #print tensor_a._shape

    tensor_b = Conv2D(paras[1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    tensor_b1 = Conv2D(paras[2], kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(tensor_b)
    tensor_b2 = Conv2D(paras[2], kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(tensor_b)
    #print tensor_b._shape

    tensor_c = Conv2D(paras[3], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    tensor_c = Conv2D(paras[4], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(tensor_c)
    tensor_c1 = Conv2D(paras[5], kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(tensor_c)
    tensor_c2 = Conv2D(paras[5], kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(tensor_c)
    #print tensor_c._shape

    tensor_d = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    tensor_d = Conv2D(paras[6], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(tensor_d)
    #print tensor_d._shape

    output_tensor = concatenate([tensor_a,tensor_b1,tensor_b2,tensor_c1,tensor_c2,tensor_d])
    #print output_tensor._shape
    return output_tensor
def getGoogleNet_v3() :
    # https://arxiv.org/pdf/1512.00567.pdf
    #local_response_normalization = Lambda(lambda x : tf.nn.local_response_normalization(x, depth_radius=2, bias=2, alpha=1e-4, beta=0.75))
    # http://blog.csdn.net/mao_xiao_feng/article/details/53488271
    # https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization

    input_img = Input(shape=Model_Feature['InputShape'], dtype='float32', name = 'input_img')
    #299*299*3

    t = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(input_img)
    t = Conv2D(32, kernel_size=(3, 3), activation='relu')(t)
    t = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(t)
    print t._shape
    # (?, 147, 147, 64)

    #t = local_response_normalization(t)
    t = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(t)
    print t._shape
    # (?, 73, 73, 64)

    t = Conv2D(80, kernel_size=(1, 1), activation='relu')(t)
    t = Conv2D(192, kernel_size=(3, 3), activation='relu')(t)
    print t._shape
    # (?, 71, 71, 192)

    t = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(t)
    #t = local_response_normalization(t)
    print t._shape
    # (?, 35, 35, 192)
    
    t = inceptionv2_1(t, paras=[64,48,64,64,96,96,32])
    print t._shape
    # (?, 35, 35, 256)

    t = inceptionv2_1(t, paras=[64,48,64,64,96,96,64])
    t = inceptionv2_1(t, paras=[64,48,64,64,96,96,64])
    print t._shape
    # (?, 35, 35, 288)

    t = inception_pad1(t, paras=[384, 64, 96])
    #t = local_response_normalization(t)
    print t._shape
    # (?, 17, 17, 768)

    t = inceptionv2_2(t, paras=[192,128,128,192,128,128,128,128,192,192])
    print t._shape
    # (?, 17, 17, 768)

    t = inceptionv2_2(t, paras=[192,160,160,192,160,160,160,160,192,192])
    t = inceptionv2_2(t, paras=[192,160,160,192,160,160,160,160,192,192])
    print t._shape
    # (?, 17, 17, 768)

    t = inceptionv2_2(t, paras=[192,192,192,192,192,192,192,192,192,192])
    print t._shape
    # (?, 17, 17, 768)

    result_1 = getGooleNetOut_v2(t)
    print result_1._shape

    t = inception_pad2(t, paras=[192, 320, 192, 192, 192, 192])
    #t = local_response_normalization(t)
    print t._shape
    # (?, 8, 8, 1280)

    t = inceptionv2_3(t, paras=[320,384,384,488,384,384,192])
    t = inceptionv2_3(t, paras=[320,384,384,488,384,384,192])
    print t._shape
    # (?, 8, 8, 2048)

    t = AveragePooling2D(pool_size=(8, 8))(t)    
    print t._shape
    # (?, 1, 1, 2048)

    t = Flatten()(t)
    t = Dropout(0.2)(t)

    t = Dense(Model_Feature['num_classes'], activation='softmax')(t)
    print t._shape
    # (?, 17)

    optimizer = keras.optimizers.RMSprop(lr=0.004, rho=0.9, epsilon=1e-06)
    model = Model(inputs=input_img, outputs=[t, result_1])
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'], loss_weights=[1., .4])
    return model





if __name__ == '__main__':
    #getModel()
    #getModel_AlexNet()
    #getModel_VGG13()
    #getResNet50()
    #getResNet34()
    #getResNet18()
    #getGoogleNet_v1()
    getGoogleNet_v3()
    pass
    
