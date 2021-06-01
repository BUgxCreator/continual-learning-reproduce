import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    '''@Author:defeng
        copy.deepcopy is to "deep copy" some variable object, e.g., List.
        see for details: https://www.zhihu.com/question/326220443/answer/698031196
    '''

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)
    '''@Author:defeng
        we can interpret from the iteration "for" here that one run of the main.py
        is equal to multiple runs of the trainer.py(i.e., multiple experiments).
        *it means that we can set one config.json to run multiple exps.*
        because of this, multiple lines in this project is written as iteration, 
        e.g., "for device in device_type:" in function _set_device.
    '''


def _train(args):
    logfilename = '{}_{}_{}_{}_{}_{}_{}'.format(args['prefix'], args['seed'], args['model_name'], args['convnet_type'],
                                                args['dataset'], args['init_cls'], args['increment'])
    '''@Author:defeng
        {
        "prefix": "reproduce",
        "dataset": "cifar100",
        "memory_size": 2000,
        "memory_per_class": 20,
        "fixed_memory": true,
        "shuffle": true,
        "init_cls": 50,
        "increment": 10, #increase $increment classes each task. see "# Grouped accuracy" in toolkit.py
        "model_name": "UCIR",
        "convnet_type": "cosine_resnet32",
        "device": ["0"],
        "seed": [30]
        }

    '''
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    '''@Author:defeng
        see for details: https://www.cnblogs.com/xianyulouie/p/11041777.html
        26 May 2021 (Wednesday)
        format: %(filename)s enables output like this "2021-05-26 22:01:34,371 [*ucir.py*]" and we can know which file \
        a certain output come from.
    '''

    '''@Author:defeng
        set random seed and cuda devices
    '''
    _set_random()
    _set_device(args)
    print_args(args)

    '''@Author:defeng
        *set: dataset and model.*
    '''
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'])
    model = factory.get_model(args['model_name'], args)
    '''@Author:defeng
        the actual work for getting model ready is done by the .py files in the "models" folder.
    '''

    '''@Author:defeng
        cnn: softmax prediction
        nme: nearest-mean-of-neightbors prediction
        see ucir paper "Baselines" for detail.
    '''
    cnn_curve, nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))

        model.incremental_train(data_manager) #train
        cnn_accy, nme_accy = model.eval_task() #val
        model.after_task()#post-processing

        if nme_accy is not None:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            logging.info('NME: {}'.format(nme_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            nme_curve['top1'].append(nme_accy['top1'])
            nme_curve['top5'].append(nme_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}'.format(cnn_curve['top5']))
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
            logging.info('NME top5 curve: {}\n'.format(nme_curve['top5']))
        else:
            logging.info('No NME accuracy.')
            logging.info('CNN: {}'.format(cnn_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}\n'.format(cnn_curve['top5']))


def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    '''@Author:defeng
        for reproductivity.
        see for details: https://www.zhihu.com/question/345043149/answer/1634128300
        inside th link, the seeds for python random and numpy package are also set.
    '''


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
