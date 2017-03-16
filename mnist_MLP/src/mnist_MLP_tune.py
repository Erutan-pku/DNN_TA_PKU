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
22 0.9837
27 0.9832 0.9834
1
13 0.9843
27 0.9836 0.9825
2
10 0.9845
25 0.9847 0.9832
3
26 0.9832
23 0.9844 0.9822
4
19 0.9848
23 0.9844 0.9828
5
28 0.9842
29 0.9842 0.9838
6
23 0.9844
26 0.9836 0.9834
7
29 0.9841
27 0.9840 0.9829
8
29 0.9843
25 0.9842 0.9835
9
29 0.9839
18 0.9841 0.9829
64 1
28 0.9761
26 0.9767 0.9750
64 2
22 0.9758
26 0.9769 0.9734
64 3
23 0.9745
28 0.9760 0.9737
64 4
20 0.9734
20 0.9758 0.9734
64 5
23 0.9749
25 0.9746 0.9732
64 6
24 0.9723
22 0.9751 0.9710
128 1
29 0.9812
23 0.9791 0.9784
128 2
20 0.9816
17 0.9815 0.9813
128 3
19 0.9802
29 0.9809 0.9794
128 4
26 0.9808
26 0.9817 0.9808
128 5
24 0.9805
24 0.9816 0.9805
128 6
24 0.9799
19 0.9816 0.9777
256 1
17 0.9840
26 0.9832 0.9818
256 2
19 0.9840
23 0.9836 0.9831
256 3
26 0.9828
28 0.9831 0.9821
256 4
23 0.9840
26 0.9837 0.9824
256 5
29 0.9818
11 0.9814 0.9810
256 6
25 0.9809
27 0.9814 0.9796
512 1
27 0.9843
29 0.9842 0.9836
512 2
22 0.9837
26 0.9845 0.9817
512 3
28 0.9842
26 0.9834 0.9818
512 4
18 0.9837
24 0.9810 0.9816
512 5
22 0.9835
29 0.9813 0.9812
512 6
10 0.9772
8 0.9783 0.9743
1024 1
21 0.9834
13 0.9832 0.9830
1024 2
24 0.9837
24 0.9837 0.9837
1024 3
19 0.9831
28 0.9834 0.9828
1024 4
28 0.9819
25 0.9808 0.9795
1024 5
7 0.9796
7 0.9779 0.9796
1024 6
4 0.9716
4 0.9723 0.9716
[512, 256, 128]
23 0.9838
27 0.9838 0.9817
[128, 256, 512]
24 0.9802
25 0.9811 0.9795
[256, 128, 64]
25 0.9834
9 0.9827 0.9831
[64, 128, 256]
19 0.9765
18 0.9758 0.9726
[512, 128, 32, 10]
29 0.9819
14 0.9806 0.9791
[512, 128, 32, 5]
26 0.9699
26 0.9718 0.9699
softplus
27 0.9842
24 0.9835 0.9830
relu
22 0.9836
20 0.9843 0.9825
tanh
26 0.9828
29 0.9826 0.9803
sigmoid
29 0.9848
26 0.9834 0.9828
hard_sigmoid
29 0.9829
23 0.9818 0.9822
0
10 0.9834
28 0.9819 0.9815
0.1
27 0.9841
11 0.9833 0.9828
0.2
19 0.9855
21 0.9841 0.9851
0.3
20 0.9843
21 0.9840 0.9835
0.5
18 0.9838
20 0.9833 0.9832
0 <keras.optimizers.SGD object at 0x7f5cebff7890>
29 0.9451
29 0.9504 0.9451
29 0.9451
29 0.9504 0.9451
0 <keras.optimizers.RMSprop object at 0x7f5cebf79dd0>
22 0.9843
19 0.9843 0.9829
22 0.9843
19 0.9843 0.9829
0 <keras.optimizers.Adagrad object at 0x7f5cebfea1d0>
28 0.9833
28 0.9823 0.9833
28 0.9833
28 0.9823 0.9833
0 <keras.optimizers.Adadelta object at 0x7f5cebb41c50>
25 0.9831
26 0.9818 0.9821
25 0.9831
26 0.9818 0.9821
0 <keras.optimizers.Adam object at 0x7f5cebea7ed0>
27 0.9845
26 0.9850 0.9836
27 0.9845
26 0.9850 0.9836
0 <keras.optimizers.Adamax object at 0x7f5cebbaeb50>
29 0.9852
27 0.9835 0.9825
29 0.9852
27 0.9835 0.9825
1 <keras.optimizers.SGD object at 0x7f5cebe5ad90>
29 0.9616
29 0.9651 0.9616
29 0.9616
29 0.9651 0.9616
1 <keras.optimizers.RMSprop object at 0x7f5ce9862b10>
16 0.9843
27 0.9840 0.9841
16 0.9843
27 0.9840 0.9841
1 <keras.optimizers.Adagrad object at 0x7f5ce97cfc10>
18 0.9848
28 0.9831 0.9847
18 0.9848
28 0.9831 0.9847
1 <keras.optimizers.Adadelta object at 0x7f5ce9798150>
26 0.9855
28 0.9835 0.9841
26 0.9855
28 0.9835 0.9841
1 <keras.optimizers.Adam object at 0x7f5ce9440d50>
25 0.9838
24 0.9834 0.9835
25 0.9838
24 0.9834 0.9835
1 <keras.optimizers.Adamax object at 0x7f5ce9468bd0>
26 0.9853
25 0.9837 0.9833
26 0.9853
25 0.9837 0.9833
2 <keras.optimizers.SGD object at 0x7f5ce9750510>
29 0.9736
29 0.9741 0.9736
29 0.9736
29 0.9741 0.9736
2 <keras.optimizers.RMSprop object at 0x7f5cd5172ed0>
23 0.9834
16 0.9830 0.9820
23 0.9834
16 0.9830 0.9820
2 <keras.optimizers.Adagrad object at 0x7f5cd51d2450>
21 0.9847
27 0.9836 0.9830
21 0.9847
27 0.9836 0.9830
2 <keras.optimizers.Adadelta object at 0x7f5cd50fae50>
13 0.9847
29 0.9843 0.9841
13 0.9847
29 0.9843 0.9841
2 <keras.optimizers.Adam object at 0x7f5cd50edb10>
14 0.9839
25 0.9832 0.9836
14 0.9839
25 0.9832 0.9836
2 <keras.optimizers.Adamax object at 0x7f5cd50b6f10>
23 0.9846
14 0.9837 0.9826
23 0.9846
14 0.9837 0.9826
3 <keras.optimizers.SGD object at 0x7f5cd5044d90>
29 0.9789
28 0.9790 0.9788
29 0.9789
28 0.9790 0.9788
3 <keras.optimizers.RMSprop object at 0x7f5cab3cae10>
28 0.9796
27 0.9818 0.9792
28 0.9796
27 0.9818 0.9792
3 <keras.optimizers.Adagrad object at 0x7f5cab3d8490>
0 0.0958
0 0.0967 0.0958
0 0.0958
0 0.0967 0.0958
3 <keras.optimizers.Adadelta object at 0x7f5caafa3090>
27 0.9852
14 0.9847 0.9833
27 0.9852
14 0.9847 0.9833
3 <keras.optimizers.Adam object at 0x7f5cab2e01d0>
19 0.9804
28 0.9802 0.9797
19 0.9804
28 0.9802 0.9797
3 <keras.optimizers.Adamax object at 0x7f5cab2fab50>
29 0.9856
25 0.9845 0.9852
29 0.9856
25 0.9845 0.9852
4 <keras.optimizers.SGD object at 0x7f5cab288c50>
29 0.9823
29 0.9819 0.9823
29 0.9823
29 0.9819 0.9823
4 <keras.optimizers.RMSprop object at 0x7f5ca8c88c50>
27 0.9697
27 0.9710 0.9697
27 0.9697
27 0.9710 0.9697
4 <keras.optimizers.Adagrad object at 0x7f5ca8c31bd0>
0 0.1032
0 0.0990 0.1032
0 0.1032
0 0.0990 0.1032
4 <keras.optimizers.Adadelta object at 0x7f5ca889e410>
20 0.9839
27 0.9857 0.9830
20 0.9839
27 0.9857 0.9830
4 <keras.optimizers.Adam object at 0x7f5ca8c28f10>
26 0.9740
26 0.9763 0.9740
26 0.9740
26 0.9763 0.9740
4 <keras.optimizers.Adamax object at 0x7f5ca88f4b10>
27 0.9820
24 0.9830 0.9808
27 0.9820
24 0.9830 0.9808
5 <keras.optimizers.SGD object at 0x7f5ca8beed10>
29 0.9840
29 0.9833 0.9840
29 0.9840
29 0.9833 0.9840
5 <keras.optimizers.RMSprop object at 0x7f5c06e84dd0>
0 0.1028
0 0.1090 0.1028
0 0.1028
0 0.1090 0.1028
5 <keras.optimizers.Adagrad object at 0x7f5c06e60a10>
29 0.1028
29 0.1090 0.1028
29 0.1028
29 0.1090 0.1028
5 <keras.optimizers.Adadelta object at 0x7f5c06ac1950>
28 0.9842
20 0.9834 0.9839
28 0.9842
20 0.9834 0.9839
5 <keras.optimizers.Adam object at 0x7f5c06ae5490>
21 0.9560
26 0.9611 0.9506
21 0.9560
26 0.9611 0.9506
5 <keras.optimizers.Adamax object at 0x7f5c06b39c90>
21 0.9769
29 0.9793 0.9745
21 0.9769
29 0.9793 0.9745
20 0.9601
20 0.9609 0.9601
0
29 0.9840
29 0.9832 0.9840
29 0.9840
29 0.9832 0.9840
1
23 0.9843
25 0.9844 0.9840
23 0.9843
25 0.9844 0.9840
2
14 0.9844
17 0.9844 0.9820
14 0.9844
17 0.9844 0.9820
3
22 0.9840
29 0.9849 0.9826
22 0.9840
29 0.9849 0.9826
4
29 0.9846
29 0.9843 0.9846
29 0.9846
29 0.9843 0.9846
5
27 0.9847
13 0.9835 0.9840
27 0.9847
13 0.9835 0.9840
6
15 0.9850
26 0.9850 0.9835
15 0.9850
26 0.9850 0.9835
7
22 0.9858
27 0.9846 0.9825
22 0.9858
27 0.9846 0.9825
8
22 0.9845
16 0.9834 0.9830
22 0.9845
16 0.9834 0.9830
9
10 0.9843
27 0.9839 0.9823
10 0.9843
27 0.9839 0.9823

"""