#coding=utf-8
#-*- coding: UTF-8 -*- 
import sys
import numpy as np
from mnist_MLP import minist_MLP, mnist_MLP_Parameters
from keras.optimizers import *

def resultAnalisys(result):
    nb_epoch = len(result['test_result'])
    best_test_acc = -1
    best_test_id  = 0
    for i, test_i in enumerate(result['test_result']) :
        if test_i['test_acc'] > best_test_acc :
            best_test_acc = test_i['test_acc']
            best_test_id = i
    best_val_acc = -1
    best_val_id  = 0
    for i, history_i in enumerate(result['history']) :
        if history_i['val_acc'][0] > best_val_acc :
            best_val_acc = history_i['val_acc'][0]
            best_val_id = i
    choose_test_acc = result['test_result'][best_val_id]['test_acc']

    print best_test_id, '%.4f'%(best_test_acc)
    print best_val_id, '%.4f'%(best_val_acc), '%.4f'%(choose_test_acc)
    return {
        'best_test':{'best_test_id':best_test_id, 'best_test_acc':best_test_acc},
        'best_val':{'best_val_id':best_val_id, 'best_val_acc':best_val_acc, 'choose_test_acc':choose_test_acc}
    }



if __name__ == '__main__':
    para = mnist_MLP_Parameters
    """
    for s in [64, 128, 256, 512, 1024] :
        for t in [1,2,3,4,5,6] :
            if s > 256 and t > 4 :
                continue
            print s, t
            para['model_layers']['denseSize'] = [s] * t
            result = minist_MLP(para)
            resultAnalisys(result)
    para['model_layers']['denseSize'] = [512,256,128]
    result = minist_MLP(para)
    resultAnalisys(result)
    28 0.9842
    28 0.9838 0.9842
    """
    """
    for activation in ['softplus', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid'] :
        print activation
        para['model_layers']['activation'] = activation
        result = minist_MLP(para)
        resultAnalisys(result)
    para['model_layers']['activation'] = 'relu'
    for dropout in [0, 0.1, 0.2, 0.3, 0.5] :
        print dropout
        para['model_layers']['dropout'] = dropout
        result = minist_MLP(para)
        resultAnalisys(result)
    """
    #[SGD(lr=0.01), RMSprop(lr=0.001), Adagrad(lr=0.01), Adadelta(lr=1.0), Adam(lr=0.001), Adamax(lr=0.002)]
    lr_list = np.array([0.01, 0.001, 0.01, 1.0, 0.001, 0.002])
    for i in range(5) :
        for optimizer in [SGD(lr=lr_list[0]), RMSprop(lr=lr_list[1]), Adagrad(lr=lr_list[2]), Adadelta(lr=lr_list[3]), Adam(lr=lr_list[4]), Adamax(lr=lr_list[5])] :
            print i, optimizer
            para['optimizer'] = optimizer
            result = minist_MLP(para)
            resultAnalisys(result)
        lr_list *= 2

"""
学习速率、正则化项、随机优化算法。

激活函数选择

隐藏层数量、隐藏节点个数
64 1
25 0.9752
29 0.9762 0.9746
64 2
23 0.9755
27 0.9767 0.9738
64 3
20 0.9744
28 0.9770 0.9741
64 4
27 0.9746
27 0.9765 0.9746
64 5
19 0.9730
26 0.9742 0.9723
64 6
28 0.9720
29 0.9730 0.9699
128 1
25 0.9799
24 0.9817 0.9780
128 2
29 0.9829
18 0.9806 0.9803
128 3
20 0.9808
15 0.9810 0.9793
128 4
10 0.9796
22 0.9815 0.9778
128 5
19 0.9785
26 0.9808 0.9770
128 6
21 0.9797
28 0.9805 0.9787
256 1
22 0.9830
23 0.9834 0.9817
256 2
11 0.9843
26 0.9841 0.9832
256 3
20 0.9827
27 0.9830 0.9813
256 4
16 0.9824
21 0.9814 0.9806
256 5
24 0.9829
14 0.9813 0.9797
256 6
23 0.9819
28 0.9809 0.9806
512 1
28 0.9842
26 0.9835 0.9827
512 2
18 0.9840
18 0.9835 0.9840
512 3
28 0.9836
26 0.9835 0.9833
512 4
27 0.9830
20 0.9819 0.9794
1024 1
14 0.9846
25 0.9840 0.9830
1024 2
19 0.9853
18 0.9835 0.9822
1024 3
29 0.9827
26 0.9834 0.9824
1024 4
26 0.9819
17 0.9815 0.9803

softplus
28 0.9848
29 0.9830 0.9825
relu
26 0.9845
28 0.9829 0.9840
tanh
29 0.9839
27 0.9822 0.9812
sigmoid
28 0.9843
25 0.9830 0.9830
hard_sigmoid
27 0.9829
28 0.9823 0.9824
0
13 0.9826
13 0.9818 0.9826
0.1
20 0.9842
25 0.9834 0.9840
0.2
7 0.9836
24 0.9839 0.9835
0.3
28 0.9844
28 0.9836 0.9844
0.5
15 0.9832
22 0.9829 0.9816


<keras.optimizers.SGD object at 0x7f7eceb84c90>
77 0.9764
76 0.9781 0.9757
<keras.optimizers.RMSprop object at 0x7f680b940390>
... loading data
60 0.9854
68 0.9854 0.9835
<keras.optimizers.Adagrad object at 0x7f680b8dbc10>
... loading data
66 0.9869
39 0.9852 0.9857
<keras.optimizers.Adadelta object at 0x7f680b905410>
... loading data
50 0.9848
76 0.9857 0.9836
<keras.optimizers.Adam object at 0x7f680b894e10>
... loading data
39 0.9854
64 0.9845 0.9827
<keras.optimizers.Adamax object at 0x7f680b85aed0>
... loading data
62 0.9855
50 0.9856 0.9851
<keras.optimizers.SGD object at 0x7f7eceb84c90>
79 0.9816
77 0.9812 0.9813
<keras.optimizers.RMSprop object at 0x7fefcca95390>
... loading data
68 0.9845
68 0.9855 0.9845
<keras.optimizers.Adagrad object at 0x7fefcca32c10>
... loading data
60 0.9851
49 0.9846 0.9839
<keras.optimizers.Adadelta object at 0x7fefcca5b410>
... loading data
76 0.9862
57 0.9857 0.9850
<keras.optimizers.Adam object at 0x7fefcc9ebe10>
... loading data
48 0.9843
59 0.9844 0.9823
<keras.optimizers.Adamax object at 0x7fefcc9b0ed0>
... loading data
21 0.9861
57 0.9856 0.9849

"""