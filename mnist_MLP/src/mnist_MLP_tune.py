#coding=utf-8
#-*- coding: UTF-8 -*- 
import sys
import codecs
import numpy as np
from mnist_MLP import minist_MLP, mnist_MLP_Parameters
from keras.regularizers import l2
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
        'best_val':{'best_val_id':best_val_id, 'best_val_acc':best_val_acc, 'choose_test_acc':1-choose_test_acc}
    }

if __name__ == '__main__':
    para = mnist_MLP_Parameters
    output = codecs.open('mnist_MLP_tune.result', "w", "utf-8")
    
    #"""
    list_1 = np.zeros(10)
    for i in range(10) :
        print i
        result = minist_MLP(para)
        list_1[i] = resultAnalisys(result)['best_val']['choose_test_acc']
    output.write(str(list_1)+'\n\n')
    output.flush()

    matrix_1 = np.zeros([5, 6])
    for i, s in enumerate([64, 128, 256, 512, 1024]) :
        for j, t in enumerate([1,2,3,4,5,6]) :
            print s, t
            para['model_layers']['denseSize'] = [s] * t
            result = minist_MLP(para)
            matrix_1[i][j] = resultAnalisys(result)['best_val']['choose_test_acc']
    output.write(str(matrix_1)+'\n\n')
    output.flush()
    list_2 = np.zeros(6)
    for i, st in enumerate([[512,256,128],[128,256,512],[256,128,64],[64,128,256],[512,128,32,10],[512,128,32,5]]) :
        print st
        para['model_layers']['denseSize'] = st
        result = minist_MLP(para)
        list_2[i] = resultAnalisys(result)['best_val']['choose_test_acc']
    para['model_layers']['denseSize']=[512,512]
    output.write(str(list_2)+'\n\n')
    output.flush()

    list_3 = np.zeros(5)
    for i, activation in enumerate(['softplus', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid']) :
        print activation
        para['model_layers']['activation'] = activation
        result = minist_MLP(para)
        list_3[i] = resultAnalisys(result)['best_val']['choose_test_acc']
    para['model_layers']['activation'] = 'relu'
    output.write(str(list_3)+'\n\n')
    output.flush()

    list_4 = np.zeros(5)
    for i, dropout in enumerate([0, 0.1, 0.2, 0.3, 0.5]) :
        print dropout
        para['model_layers']['dropout'] = dropout
        result = minist_MLP(para)
        list_4[i] = resultAnalisys(result)['best_val']['choose_test_acc']
    para['model_layers']['dropout'] = 0.2
    output.write(str(list_4)+'\n\n')
    output.flush()

    #[SGD(lr=0.01), RMSprop(lr=0.001), Adagrad(lr=0.01), Adadelta(lr=1.0), Adam(lr=0.001), Adamax(lr=0.002)]
    matrix_2 = np.zeros([6, 6])
    matrix_3 = np.zeros([6, 6])
    lr_list = np.array([0.01, 0.001, 0.01, 1.0, 0.001, 0.002])
    lr_list /= 2
    for i in range(6) :
        for j, optimizer in enumerate([SGD(lr=lr_list[0]), RMSprop(lr=lr_list[1]), Adagrad(lr=lr_list[2]), Adadelta(lr=lr_list[3]), Adam(lr=lr_list[4]), Adamax(lr=lr_list[5])]) :
            print i, optimizer
            para['optimizer'] = optimizer
            result = minist_MLP(para)
            matrix_2[i][j] = resultAnalisys(result)['best_val']['choose_test_acc']
            matrix_3[i][j] = resultAnalisys(result)['best_val']['best_val_id']
        lr_list *= 2
    para['optimizer'] = RMSprop()
    output.write(str(matrix_2)+'\n\n')
    output.write(str(matrix_3)+'\n\n')
    output.flush()

    para['model_layers']['W_regularizer'] = l2(0.01)
    result = minist_MLP(para)
    vt = resultAnalisys(result)['best_val']['choose_test_acc']
    output.write(str(vt)+'\n\n')
    para['model_layers']['W_regularizer'] = None

    list_5 = np.zeros(10)
    for i in range(10) :
        print i
        result = minist_MLP(para)
        resultAnalisys(result)
        list_5[i] = resultAnalisys(result)['best_val']['choose_test_acc']
    output.write(str(list_5)+'\n\n')
    output.flush()
    output.close()
    #"""
"""
正则化项

学习速率、随机优化算法。

激活函数选择

隐藏层数量、隐藏节点个数
512-256-128
28 0.9842
28 0.9838 0.9842
128-256-512
28 0.9807
19 0.9825 0.9779
256-128-64
27 0.9842
27 0.9844 0.9842
64-128-256
18 0.9848
15 0.9835 0.9831
512-128-32-8
25 0.9837
24 0.9832 0.9827
512-128-32-8-4
26 0.9846
26 0.9838 0.9846


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
512 5
27 0.9816
24 0.9812 0.9786
512 6
6 0.9793
6 0.9796 0.9793
1024 5
5 0.9782
16 0.9789 0.9779
1024 6
8 0.9846
19 0.9833 0.9834

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


0 <keras.optimizers.SGD object at 0x7f06c35b1a90>
29 0.9617
29 0.9647 0.9617
0 <keras.optimizers.RMSprop object at 0x7f06c35ca590>
20 0.9841
23 0.9835 0.9816
0 <keras.optimizers.Adagrad object at 0x7f06c3567c90>
23 0.9843
20 0.9823 0.9832
0 <keras.optimizers.Adadelta object at 0x7f06c358f490>
29 0.9843
23 0.9841 0.9829
0 <keras.optimizers.Adam object at 0x7f06c351ce90>
29 0.9841
17 0.9843 0.9825
0 <keras.optimizers.Adamax object at 0x7f06c36009d0>
22 0.9862
22 0.9834 0.9862
1 <keras.optimizers.SGD object at 0x7f06c3556350>
29 0.9721
28 0.9744 0.9714
1 <keras.optimizers.RMSprop object at 0x7f0650357850>
24 0.9834
16 0.9826 0.9813
1 <keras.optimizers.Adagrad object at 0x7f06502dd910>
20 0.9844
24 0.9839 0.9829
1 <keras.optimizers.Adadelta object at 0x7f064fed1a90>
25 0.9849
21 0.9847 0.9846
1 <keras.optimizers.Adam object at 0x7f064feed4d0>
26 0.9844
14 0.9832 0.9826
1 <keras.optimizers.Adamax object at 0x7f064fefbf10>
26 0.9857
24 0.9857 0.9853
2 <keras.optimizers.SGD object at 0x7f064ff23fd0>
28 0.9798
27 0.9800 0.9786
2 <keras.optimizers.RMSprop object at 0x7f06469dbcd0>
23 0.9787
16 0.9802 0.9774
2 <keras.optimizers.Adagrad object at 0x7f0646993950>
0 0.1032
0 0.0990 0.1032
2 <keras.optimizers.Adadelta object at 0x7f064656e750>
16 0.9864
24 0.9845 0.9834
2 <keras.optimizers.Adam object at 0x7f0646881450>
29 0.9811
27 0.9800 0.9793
2 <keras.optimizers.Adamax object at 0x7f064653cdd0>
17 0.9858
27 0.9840 0.9854
3 <keras.optimizers.SGD object at 0x7f0646871e10>
29 0.9836
26 0.9825 0.9826
3 <keras.optimizers.RMSprop object at 0x7f0644281f90>
22 0.9738
24 0.9724 0.9702
3 <keras.optimizers.Adagrad object at 0x7f06442d9f10>
0 0.0982
0 0.0983 0.0982
3 <keras.optimizers.Adadelta object at 0x7f06441e3110>
26 0.9854
27 0.9847 0.9839
3 <keras.optimizers.Adam object at 0x7f0643e89d10>
... loading data
27 0.9742
25 0.9751 0.9733
3 <keras.optimizers.Adamax object at 0x7f064419bf90>
... loading data
20 0.9838
29 0.9830 0.9837
4 <keras.optimizers.SGD object at 0x7f0644162e50>
... loading data
26 0.9842
24 0.9830 0.9827
4 <keras.optimizers.RMSprop object at 0x7f0641b5db10>
... loading data
0 0.1009
0 0.0961 0.1009
4 <keras.optimizers.Adagrad object at 0x7f064178cc10>
0 0.1028
0 0.1090 0.1028
4 <keras.optimizers.Adadelta object at 0x7f06417a8850>
15 0.9839
27 0.9835 0.9837
4 <keras.optimizers.Adam object at 0x7f0641ad3f10>
23 0.9600
14 0.9628 0.9567
4 <keras.optimizers.Adamax object at 0x7f0641ab7750>
27 0.9755
26 0.9762 0.9738
icstpie@gpu-srv:~/lyx/DNN_pku/mnist_MLP/src$



22 0.9845
18 0.9830 0.9817
... loading data
19 0.9846
19 0.9836 0.9846
... loading data
27 0.9839
24 0.9840 0.9827
... loading data
29 0.9840
21 0.9841 0.9832
... loading data
24 0.9844
23 0.9843 0.9831
... loading data
28 0.9838
14 0.9835 0.9811
... loading data
29 0.9840
25 0.9851 0.9832
... loading data
24 0.9838
17 0.9828 0.9832
... loading data
22 0.9848
10 0.9833 0.9826
... loading data
16 0.9849
18 0.9844 0.9830

"""