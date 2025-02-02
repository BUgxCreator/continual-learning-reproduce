import math
import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import tensor2numpy, target2onehot

EPSILON = 1e-8

# ImageNet1000, ResNet18
'''
epochs = 90
lrate = 0.1
milestones = [30, 60]
lrate_decay = 0.1
batch_size = 128
lamda_base = 10
K = 2
margin = 0.5
weight_decay = 1e-4
num_workers = 16
'''

# CIFAR100, ResNet32
epochs = 160
lrate = 0.1
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 128
lamda_base = 5
K = 2
margin = 0.5
weight_decay = 5e-4
num_workers = 4


class UCIR(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = CosineIncrementalNet(args['convnet_type'], pretrained=False)
        self._class_means = None

    def after_task(self):
        # self.save_checkpoint()
        self._old_network = self._network.copy().freeze() # from ucir paper, easy to understand.
        self._known_classes = self._total_classes # _known_classes is the old_classes which the model have trained on.
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager): #external call in trainer.py
        self._cur_task += 1 # initial cur_task=-1. first time call incremental_train, it will add 1 and become zero.
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)#total-known = task_size
        self._network.update_fc(self._total_classes, self._cur_task) #update based on new class nb.
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes)) #total-known = task_size

        # Loader
        train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                              mode='train', appendent=self._get_memory())
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Procedure
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class) # see base.py
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
            '''@Author:defeng
                the reason call .module: https://www.zhihu.com/question/67726969/answer/511220696
            '''

    def _train(self, train_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''

        # Adaptive lambda
        # The definition of adaptive lambda in paper and the official code repository is different.
        # Here we use the definition in official code repository.
        if self._cur_task == 0:
            self.lamda = 0
        else:
            self.lamda = lamda_base * math.sqrt(self._known_classes / (self._total_classes - self._known_classes))
        logging.info('Adaptive lambda: {}'.format(self.lamda))

        # Fix the embedding of old classes
        if self._cur_task == 0:
            network_params = self._network.parameters()
        else:
            ignored_params = list(map(id, self._network.fc.fc1.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, self._network.parameters())
            network_params = [{'params': base_params, 'lr': lrate, 'weight_decay': weight_decay},
                              {'params': self._network.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}] # lr=0 means freeze.
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
            '''@Author:defeng
                in base.py, "self._device = args['device'][0]"
                that is, the old model is moved to device[0] default.
            '''

        self._run(train_loader, test_loader, optimizer, scheduler)

    def _run(self, train_loader, test_loader, optimizer, scheduler):
        for epoch in range(1, epochs+1):
            self._network.train() # set train mode
            ce_losses = 0.
            lf_losses = 0.
            is_losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs['logits']  # Final outputs after scaling  (bs, nb_classes), |* i.e., befroe probs=softmax(logits)
                features = outputs['features']  # Features before fc layer  (bs, 64) |* i.e., feature vector from feature extractor(backbone)
                ce_loss = F.cross_entropy(logits, targets)  # Cross entropy loss |* cross_entrophy implicityly implement softmax, so its input is logits.

                lf_loss = 0.  # Less forgetting loss
                is_loss = 0.  # Inter-class speration loss, i.e. margin ranking loss. Eq 8.
                if self._old_network is not None:
                    old_outputs = self._old_network(inputs)
                    old_features = old_outputs['features']  # Features before fc layer
                    lf_loss = F.cosine_embedding_loss(features, old_features.detach(),
                                                      torch.ones(inputs.shape[0]).to(self._device)) * self.lamda # Eq 6.

                    scores = outputs['new_scores']  # Scores before scaling  (bs, nb_new)
                    old_scores = outputs['old_scores']  # Scores before scaling  (bs, nb_old)
                    '''@Author:defeng
                        24 May 2021 (Monday)
                        see Line 45 here, we know ucir uses CosineincNet and CosineincNet uses (Split)CosineLinearLayer.
                        Line 93 forward function of SplitCosineLinearLayer, "out" times(X) the scaling factor eta while out1/2 doesn't.
                        (CosineLinearLayer does not have the new/old_scores.)
                    '''
                    old_classes_mask = np.where(tensor2numpy(targets) < self._known_classes)[0]
                    if len(old_classes_mask) != 0:
                        scores = scores[old_classes_mask]  # (n, nb_new)
                        old_scores = old_scores[old_classes_mask]  # (n, nb_old)

                        # Ground truth targets
                        gt_targets = targets[old_classes_mask]  # (n)
                        old_bool_onehot = target2onehot(gt_targets, self._known_classes).type(torch.bool)
                        anchor_positive = torch.masked_select(old_scores, old_bool_onehot)  # *(n)*   |* i.e. select GT class correspoding scores.
                        anchor_positive = anchor_positive.view(-1, 1).repeat(1, K)  # *(n, K)*   |* i.e., <\bar{\theta}, \bar(f(x))>
                        '''@Author:defeng
                            torch.repeat is different from numpy.repeat.
                            see for details: https://pytorch.org/docs/stable/tensors.html?highlight=repeat#torch.Tensor.repeat
                        '''

                        # Top K hard
                        anchor_hard_negative = scores.topk(K, dim=1)[0]  # *(n, K)* |* i.e., <\bar{\theta_{k}}, \bar(f(x))>

                        is_loss = F.margin_ranking_loss(anchor_positive, anchor_hard_negative,
                                                        torch.ones(K).to(self._device), margin=margin)
                        '''@Author:defeng
                            here, the params "torch.ones(K).to(self._device)" for margin_ranking_loss follows the params \
                            requirements in pytorch documentation(specifically, ones(K) is the variable y).
                            see for details: https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html#torch.nn.MarginRankingLoss
                        '''

                loss = ce_loss + lf_loss + is_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ce_losses += ce_loss.item()
                lf_losses += lf_loss.item() if self._cur_task != 0 else lf_loss
                is_losses += is_loss.item() if self._cur_task != 0 and len(old_classes_mask) != 0 else is_loss

                # acc(classification)
                _, preds = torch.max(logits, dim=1) # pred is the indexs/location of the max value in dim1.
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            # train_acc = self._compute_accuracy(self._network, train_loader)
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info1 = 'Task {}, Epoch {}/{} => '.format(self._cur_task, epoch, epochs)
            info2 = 'CE_loss {:.3f}, LF_loss {:.3f}, IS_loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                ce_losses/(i+1), lf_losses/(i+1), is_losses/(i+1), train_acc, test_acc)
            logging.info(info1 + info2)
