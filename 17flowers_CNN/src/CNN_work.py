# coding=utf-8
# -*- coding: UTF-8 -*- 
import sys
from loadData import getData
import keras
import numpy as np
np.random.seed(1337)
import random
random.seed(1337)
from CNN_model import *

CNN_Feature = {
    'batch_size' : 5,
    'epochs'     : 180,
    'ignore_print_top' : 0
}

def getXY(Data) :
    X = np.array([img['img_raw'] for img in Data]) / 255.0
    Y = np.array([img['label'] for img in Data])
    Y = keras.utils.to_categorical(Y, Model_Feature['num_classes'])
    return X, Y

def trainTest(trn_data, val_data, tst_data, ModelFunction) :
    trn_x, trn_y = getXY(trn_data)
    val_x, val_y = getXY(val_data)
    tst_x, tst_y = getXY(tst_data)
    """
    print np.shape(trn_x), np.shape(trn_y)
    print np.shape(val_x), np.shape(val_y)
    print np.shape(tst_x), np.shape(tst_y)
    """

    model = ModelFunction()
    score_trn, score_val, score_tst = [], [], []
    for i in range(CNN_Feature['epochs']) :
        model.fit(trn_x, [trn_y, trn_y], batch_size=CNN_Feature['batch_size'], epochs=1, shuffle=True, verbose=1)
        if i < CNN_Feature['ignore_print_top'] :
            continue
        score_trn_i = model.evaluate(trn_x, [trn_y, trn_y], verbose=0)
        score_val_i = model.evaluate(val_x, [val_y, val_y], verbose=0)
        score_tst_i = model.evaluate(tst_x, [tst_y, tst_y], verbose=0)
        print '%.4f  %.4f'%(score_trn_i[3], score_trn_i[4])
        print '%.4f  %.4f'%(score_val_i[3], score_val_i[4])
        print '%.4f  %.4f'%(score_tst_i[3], score_tst_i[4])
        print '%d  trn_acc: %.4f  val_acc: %.4f  tst_acc: %.4f'%(i+1, score_trn_i[3], score_val_i[3], score_tst_i[3])
        score_trn.append(score_trn_i)
        score_val.append(score_val_i)
        score_tst.append(score_tst_i)
    
    max_val = max([t[3] for t in score_val])
    for i in range(CNN_Feature['epochs'] - CNN_Feature['ignore_print_top']) :
        if score_val[i][3] >= max_val :
            iter_i = CNN_Feature['ignore_print_top'] + i + 1
            max_trn = score_trn[i][3]
            max_tst = score_tst[i][3]
    print iter_i, '%.4f'%(max_tst), '%.4f'%(max_val), '%.4f'%(max_trn), '%.4f'%(max([t[1] for t in score_trn]))
    return [t[3] for t in score_val], [t[3] for t in score_tst]

if __name__ == '__main__':
    """
    for seg in [1,2,3] :
        trn_data, val_data, tst_data = getData(mode=seg, resize=Model_Feature['InputShape'][0:2])
        trn_x, trn_y = getXY(trn_data)
        val_x, val_y = getXY(val_data)
        tst_x, tst_y = getXY(tst_data)
        np.save('trn_'+str(seg)+'X.npy', trn_x)
        np.save('trn_'+str(seg)+'Y.npy', trn_y) 
        np.save('val_'+str(seg)+'X.npy', val_x)
        np.save('val_'+str(seg)+'Y.npy', val_y) 
        np.save('tst_'+str(seg)+'X.npy', tst_x)
        np.save('tst_'+str(seg)+'Y.npy', tst_y)
    #"""
    #"""
    #Models = [getModel_AlexNet, getModel_AlexNet_short, getModel_AlexNet_light, getModel_AlexNet_lighter, getModel_AlexNet_lightest]
    #infor  = ['getModel_AlexNet', 'getModel_AlexNet_short', 'getModel_AlexNet_light', 'getModel_AlexNet_lighter', 'getModel_AlexNet_lightest']
    #Models = [getModel_AlexNet_lightestzf, getModel_AlexNet_lighterzf, getModel_AlexNet_lightzf]
    #infor  = ['getModel_AlexNet_lightestzf', 'getModel_AlexNet_lighterzf', 'getModel_AlexNet_lightzf']
    #Models = [getResNet18, getResNet34, getResNet50]
    #infor  = ['getResNet18', 'getResNet34', 'getResNet50']
    Models = [getGoogleNet_v3]
    infor  = ['getGoogleNet_v3']
    for i, modelFunc in enumerate(Models) :
        print '\n\n\n'+infor[i]

        score_val_all = []
        score_tst_all = []
        for seg in [1,2,3] :
            print '\nDoing: %d'%(seg)
            trn_data, val_data, tst_data = getData(mode=seg, resize=Model_Feature['InputShape'][0:2])
            score_val_i, score_tst_i = trainTest(trn_data, val_data, tst_data, ModelFunction=modelFunc)
            score_val_all.append(score_val_i)
            score_tst_all.append(score_tst_i)
        score_val_all = [sum([score_val_all[j][i] for j in range(3)])/3 for i in range(CNN_Feature['epochs'] - CNN_Feature['ignore_print_top'])]
        score_tst_all = [sum([score_tst_all[j][i] for j in range(3)])/3 for i in range(CNN_Feature['epochs'] - CNN_Feature['ignore_print_top'])]
        max_val = max(score_val_all)
        for i, val in enumerate(score_val_all) :
            if val >= max_val :
                max_tst = score_tst_all[i]
        print '%.4f'%(max_tst), '%.4f'%(max_val)
        print '\n\n\n'
    #"""
    """
    trn_data, val_data, tst_data = getData(resize=Model_Feature['InputShape'][0:2])
    score_val_i, score_tst_i = trainTest(trn_data, val_data, tst_data, ModelFunction=getResNet50)
    #"""
    """
    hm = {
        'getResNet18' : getResNet18,
        'getResNet34' : getResNet34,
        'getResNet50' : getResNet50
    }

    print '\n\n\n'+sys.argv[1]
    score_val_all = []
    score_tst_all = []
    for seg in [1,2,3] :
        print '\nDoing: %d'%(seg)
        trn_data, val_data, tst_data = getData(mode=seg, resize=Model_Feature['InputShape'][0:2])
        score_val_i, score_tst_i = trainTest(trn_data, val_data, tst_data, ModelFunction=hm[sys.argv[1]])
        score_val_all.append(score_val_i)
        score_tst_all.append(score_tst_i)
    score_val_all = [sum([score_val_all[j][i] for j in range(3)])/3 for i in range(CNN_Feature['epochs'] - CNN_Feature['ignore_print_top'])]
    score_tst_all = [sum([score_tst_all[j][i] for j in range(3)])/3 for i in range(CNN_Feature['epochs'] - CNN_Feature['ignore_print_top'])]
    max_val = max(score_val_all)
    for i, val in enumerate(score_val_all) :
        if val >= max_val :
            max_tst = score_tst_all[i]
    print '%.4f'%(max_tst), '%.4f'%(max_val)
    print '\n\n\n'
    #"""
    

