import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CamVid
import numpy as np
import validate
import time

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
        optimizer = 'sgd'
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_count = epoch_count
        self.criterion = validate.CrossEntropyLoss2d()
        self.data_root = data_root
        self.optimizer = optimizer
        self.model_path = model_path

def train(net, train_config):
    global train_log_file
    train_log_file = open('./train.{}.log.{}.txt'.format(type(net).__name__, time.strftime("%a_%b_%d_%H_%M_%S_%Y", time.localtime())), "a")
    log('******** train begin [{}] ********'.format(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())))
    log('learning_rate: {}'.format(train_config.learning_rate))
    log('batch_size: {}'.format(train_config.batch_size))
    log('epoch_count: {}'.format(train_config.epoch_count))
    log('model_path: {}'.format(train_config.model_path))
    log('optimizer: {}'.format(train_config.optimizer))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    optimizer = None
    if train_config.optimizer == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=train_config.learning_rate)
    elif train_config.optimizer == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), train_config.learning_rate)
    elif train_config.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr = train_config.learning_rate, momentum=0.9)
    else:
        log('unkown optimizer')
        return
    train_info = []
    best_global_accuracy = None
    ### train config
    batch_size = train_config.batch_size
    epoch_count = train_config.epoch_count
    trainset = CamVid.CamVid(root=train_config.data_root, split='train')
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    valset = CamVid.CamVid(root=train_config.data_root, split='val')
    criterion = train_config.criterion
    model_path = train_config.model_path
    ### begin train
    for epoch in range(epoch_count):
        train_loss = 0.0
        iter_count = 0
        for i, data in enumerate(train_dataloader):
            inputs = data[0].to(device)
            labels = data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.squeeze(1).long())
            loss.backward()
            optimizer.step()

            iter_count += 1
            train_loss += loss.item()
            log('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
            # break
        train_loss /= iter_count

        ### validate
        global_accuracy, classes_avg_accuracy, mIoU, val_loss, classes_accuracy, classes_iou = validate.validate(net, valset, batch_size, device, criterion)

        if best_global_accuracy is None or best_global_accuracy < global_accuracy:
            log('save model')
            best_global_accuracy = global_accuracy
            torch.save(net.state_dict(), model_path)
        log("-------- Epoch #" + str(epoch + 1) + " Summary --------")
        log("mIoU: " + str(mIoU))
        log("classes_avg_accuracy: " + str(classes_avg_accuracy))
        log("global_accuracy: " + str(global_accuracy))
        log("train_loss: " + str(train_loss))
        log("val_loss: " + str(val_loss))
        train_info.append((mIoU, classes_avg_accuracy, global_accuracy, train_loss, val_loss))
        for i in range(net.num_classes):
            log('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, classes_iou[i], classes_accuracy[i], CamVid.CamVid.classes[i]))

    for epoch in range(epoch_count):
        mIoU, classes_avg_accuracy, global_accuracy, train_loss, val_loss = train_info[epoch]
        log('Epoch #%d: mIoU %.4f, classes_avg_accuracy %.4f, global_accuracy %.4f, train_loss %.4f, val_loss %.4f' % (epoch + 1, mIoU, classes_avg_accuracy, global_accuracy, train_loss, val_loss))
    
    log('******** train end [{}] ********'.format(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())))
    train_log_file.close()
