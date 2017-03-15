from mnist_MLP import minist_MLP, mnist_MLP_Parameters

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
    choose_test_acc = result['test_result']['test_acc']

    return {
        'best_test':{'best_test_id':best_test_id, 'best_test_acc':best_test_acc},
        'best_val':{'best_val_id':best_val_id, 'best_val_acc':best_val_acc, 'choose_test_acc':choose_test_acc}
    }


if __name__ == '__main__':
    para = mnist_MLP_Parameters

    result = minist_MLP(para)
    print resultAnalisys(result)