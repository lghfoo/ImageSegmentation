import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CamVid
import numpy as np
import validate
import time
import os
import errno
import math

train_log_file = None
def log(msg):
    global train_log_file
    print(msg)
    train_log_file.write(str(msg) + '\n')


class TrainConfig:
    def __init__(self, 
        learning_rate = 0.01,
        batch_size = 4,
        epoch_count = 20,
        data_root = './data/camvid',
        model_path = './model.pth',
        trained_model_path = None,
        optimizer = 'sgd',
        dataset = 'camvid11',
        split='train',
        gpu=0,
        gpus=[0,1],
        num_classes=20,
        shuffle=True
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_count = epoch_count
        self.criterion = validate.CrossEntropyLoss2d()
        self.data_root = data_root
        self.optimizer = optimizer
        self.trained_model_path = trained_model_path
        self.model_path = model_path
        self.dataset = dataset
        self.split = split
        self.gpu = gpu
        self.gpus=gpus
        self.num_classes = num_classes
        self.shuffle = shuffle


def train(net, train_config):
    global train_log_file
    train_filename = './log/train/{}.{}.txt'.format(type(net).__name__, time.strftime("%a_%b_%d_%H_%M_%S_%Y", time.localtime()))
    if not os.path.exists(os.path.dirname(train_filename)):
        try:
            os.makedirs(os.path.dirname(train_filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    train_log_file = open(train_filename, "w")
    log('******** train begin [{}] ********'.format(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())))
    log('learning_rate: {}'.format(train_config.learning_rate))
    log('batch_size: {}'.format(train_config.batch_size))
    log('epoch_count: {}'.format(train_config.epoch_count))
    log('model_path: {}'.format(train_config.model_path))
    log('trained_model_path: {}'.format(train_config.trained_model_path))
    log('optimizer: {}'.format(train_config.optimizer))
    log('dataset: {}'.format(train_config.dataset))
    log('split: {}'.format(train_config.split))
    # device = torch.device("cuda:{}".format(train_config.gpu) if torch.cuda.is_available() else "cpu")
    # net.to(device)
    net = torch.nn.DataParallel(net, device_ids=train_config.gpus).cuda()
    if train_config.trained_model_path is not None:
        net.load_state_dict(torch.load(train_config.trained_model_path))
    optimizer = None
    if train_config.optimizer == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=train_config.learning_rate)
    elif train_config.optimizer == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=train_config.learning_rate)
    elif train_config.optimizer == 'sgd' or train_config.optimizer == 'poly':
        optimizer = optim.SGD(net.parameters(), lr = train_config.learning_rate, momentum=0.9, weight_decay=0.0001)
    elif train_config.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=train_config.learning_rate)
    else:
        log('unkown optimizer')
        return
    train_info = []
    best_global_accuracy = None
    best_miou = None
    ### train config
    batch_size = train_config.batch_size
    epoch_count = train_config.epoch_count
    trainset = validate.get_dataset(train_config.dataset, train_config.split, train_config.data_root)
    valset = validate.get_dataset(train_config.dataset, 'val', train_config.data_root)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=train_config.shuffle, num_workers=0)
    criterion = train_config.criterion
    model_path = train_config.model_path
    ### begin train
    total_iter = epoch_count * (math.ceil(len(trainset) / batch_size))
    log('total_iter: {}'.format(total_iter))
    cur_iter = 0
    for epoch in range(epoch_count):
        train_loss = 0.0
        iter_count = 0
        if hasattr(net, 'training'):
            net.training = True
        for i, data in enumerate(train_dataloader):
            # inputs = data[0].to(device)
            # labels = data[1].to(device)
            inputs = data[0].cuda()
            labels = data[1].cuda()

            optimizer.zero_grad()
            outputs = net(inputs)

            
            # todo : aux loss
            loss = criterion(outputs, labels.squeeze(1).long())
            loss.backward()
            optimizer.step()
            
            iter_count += 1
            cur_iter += 1
            train_loss += loss.item()

            log('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

            if train_config.optimizer == 'poly':
                lr = train_config.learning_rate * pow(1- (cur_iter+1)/total_iter, 0.9)
                for g in optimizer.param_groups:
                    g['lr'] = lr
        if train_config.optimizer == 'poly':
            log('[%d] learning_rate: %f' % (epoch+1, train_config.learning_rate * pow(1- (cur_iter+1)/total_iter, 0.9)))

            # break
        train_loss /= iter_count

        ### validate
        if hasattr(net, 'training'):
            net.training = False
        global_accuracy, classes_avg_accuracy, mIoU, val_loss, classes_accuracy, classes_iou = validate.validate(net, valset, batch_size, train_config.gpus, criterion, num_classes=train_config.num_classes)

        if best_global_accuracy is None or best_global_accuracy < global_accuracy:
            log('save model')
            best_global_accuracy = global_accuracy
            torch.save(net.state_dict(), model_path)
        if best_miou is None or best_miou < mIoU:
            log('save model for best mIoU')
            best_miou = mIoU
            torch.save(net.state_dict(), model_path.replace('.pth', '_mIoU.pth'))
        log("-------- Epoch #" + str(epoch + 1) + " Summary --------")
        log("mIoU: " + str(mIoU))
        log("classes_avg_accuracy: " + str(classes_avg_accuracy))
        log("global_accuracy: " + str(global_accuracy))
        log("train_loss: " + str(train_loss))
        log("val_loss: " + str(val_loss))
        train_info.append((mIoU, classes_avg_accuracy, global_accuracy, train_loss, val_loss))
        for i in range(train_config.num_classes):
            log('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, classes_iou[i], classes_accuracy[i], validate.get_dataset_classes(train_config.dataset)[i]))

    for epoch in range(epoch_count):
        mIoU, classes_avg_accuracy, global_accuracy, train_loss, val_loss = train_info[epoch]
        log('Epoch #%d: mIoU %.4f, classes_avg_accuracy %.4f, global_accuracy %.4f, train_loss %.4f, val_loss %.4f' % (epoch + 1, mIoU, classes_avg_accuracy, global_accuracy, train_loss, val_loss))
    
    log('******** train end [{}] ********'.format(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())))
    train_log_file.close()
