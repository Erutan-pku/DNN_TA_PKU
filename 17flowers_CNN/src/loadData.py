# coding=utf-8
# -*- coding: UTF-8 -*- 
import sys
sys.path.append('../../public_Tools')
from IO import *
import scipy.io as sio
import numpy as np
from PIL import Image

"""
参考: http://blog.csdn.net/sinat_16823063/article/details/53946549
"""

def getData(mode=1, resize=None) :
    assert mode in [1,2,3]
    # raw_shape:(499~1057, 499~1093)
    # mask color : black:0, yellow:3, red:1
    if not resize is None :
        getPic = lambda filePath : np.array(Image.open(filePath).resize(resize))
    else :
        getPic = lambda filePath : np.array(Image.open(filePath))

    jpg_files = []
    imlist    = set(sio.loadmat('../data/trimaps/imlist.mat')['imlist'][0])
    fileName_list = loadLists('../data/jpg/files.txt')
    mask_set = set()
    for i, fileName in enumerate(fileName_list) :
        jpg_file_t = {}
        jpg_file_t['label'] = int(i / 80)
        jpg_file_t['img_raw'] = getPic('../data/jpg/'+fileName)
        #jpg_file_t['Mask'] = getPic('../data/trimaps/'+fileName[:-4]+'.png') if (i+1) in imlist else None
        jpg_files.append(jpg_file_t)

    datasplits=sio.loadmat('../data/datasplits.mat')
    keys = [tp+str(mode) for tp in ['trn', 'val', 'tst']]
    trn_set, val_set, tst_set = [set(list(datasplits[name][0])) for name in keys]
    trn_data, val_data, tst_data = [], [], []
    for i, jpg_file in enumerate(jpg_files) :
        num = i + 1
        if num in trn_set :
            trn_data.append(jpg_file)
        elif num in val_set :
            val_data.append(jpg_file)
        elif num in tst_set :
            tst_data.append(jpg_file)

    return trn_data, val_data, tst_data


if __name__ == '__main__':
    for mode in [1,2,3] :
        trn_data, val_data, tst_data = getData(mode=mode)
        print len(trn_data), len(val_data), len(tst_data)       


