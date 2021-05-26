import os
import numpy as np
import torch


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
    '''@Author:defeng
        numel stans for num of elements (in a tensor).
        see for details: https://blog.csdn.net/qq_36215315/article/details/88918260

         "if p.requires_grad" means that for those models which is in train mode,
         only the trainable(i.e.,requires_grad==True) params are computed for 
         the total number of the params of the network.
    '''


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot
    '''@Author:defeng TODO the shape of the params of the func
        I prefer using "F.one_hot(x, num_classes=C)"
        see for details: https://pytorch.org/docs/stable/nn.functional.html?highlight=one%20hot#torch.nn.functional.one_hot
    '''


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), 'Data length error.'

    all_acc = {}
    all_acc['total'] = np.around((y_pred == y_true).sum()*100 / len(y_true), decimals=2)
    '''@Author:defeng
        1. "*100" for %.
        2. function around: https://blog.csdn.net/runmin1/article/details/89174511
    '''

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment): #step for iteration is increment.
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        '''@Author:defeng
            np.where()[0] is for the reason that np.where() always return a tuple.
        '''
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+increment-1).rjust(2, '0')) #rjust: aligh right
        all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)
    '''@Author:defeng 
        Grouped acc is acc for every task, i.e., each classes in a task is viewd as a group.
        (from "y_true >= class_id, y_true < class_id + increment)" and for iteration we can know.)
    '''

    # Old accuracy(i.e., old for classes)
    idxes = np.where(y_true < nb_old)[0]
    all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes),
                                                         decimals=2)

    # New accuracy(i.e., acc for new classes)
    idxes = np.where(y_true >= nb_old)[0]
    all_acc['new'] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])
    '''@Author:defeng
        suppose imgs is like:
        [
            [img1,label1],[img2,label2],...,[imgn,labeln]
        ]
    '''

    return np.array(images), np.array(labels)
