import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CamVid
import numpy as np
import validate

class TrainConfig:
    def __init__(self, 
        learning_rate = 0.01,
        batch_size = 4,
        epoch_count = 20,
        data_root = './data/camvid',
        model_path = './model.pth'
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch_count = epoch_count
        self.criterion = validate.CrossEntropyLoss2d()
        self.data_root = data_root
        self.model_path = model_path

def train(net, train_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    optimizer = optim.Adagrad(net.parameters(), lr=train_config.learning_rate)
    train_info = []
    best_global_accuracy = None
    ### train config
    batch_size = train_config.batch_size
    epoch_count = train_config.epoch_count
    trainset = CamVid.CamVid(root=train_config.data_root, split='train')
    valset = CamVid.CamVid(root=train_config.data_root, split='val')
    criterion = train_config.criterion
    model_path = train_config.model_path
    ### begin train
    for epoch in range(epoch_count):
        i = 0
        train_loss = 0.0
        iter_count = 0
        while i + batch_size < len(trainset):
            inputs_and_labels = [trainset[i+j] for j in range(batch_size)]
            inputs = torch.stack([inputs_and_labels[i][0] for i in range(batch_size)]).to(device)
            labels = torch.stack([torch.as_tensor(np.array(inputs_and_labels[i][1])) for i in range(batch_size)]).to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.squeeze(1).long())
            loss.backward()
            optimizer.step()

            i += batch_size
            iter_count += 1
            train_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, int(i / batch_size), loss.item()))
            # break
        train_loss /= iter_count

        ### validate
        global_accuracy, classes_avg_accuracy, mIoU, val_loss, classes_accuracy, classes_iou = validate.validate(net, valset, batch_size, device, criterion)

        if best_global_accuracy is None or best_global_accuracy < global_accuracy:
            print('save model')
            best_global_accuracy = global_accuracy
            torch.save(net.state_dict(), model_path)
        print("-------- Epoch #" + str(epoch + 1) + " Summary --------")
        print("mIoU: " + str(mIoU))
        print("classes_avg_accuracy: " + str(classes_avg_accuracy))
        print("global_accuracy: " + str(global_accuracy))
        print("train_loss: " + str(train_loss))
        print("val_loss: " + str(val_loss))
        train_info.append((mIoU, classes_avg_accuracy, global_accuracy, train_loss, val_loss))
        for i in range(net.num_classes):
            print('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, classes_iou[i], classes_accuracy[i], CamVid.CamVid.classes[i]))

    for epoch in range(epoch_count):
        mIoU, classes_avg_accuracy, global_accuracy, train_loss, val_loss = train_info[epoch]
        print('Epoch #%d: mIoU %.4f, classes_avg_accuracy %.4f, global_accuracy %.4f, train_loss %.4f, val_loss %.4f' % (epoch + 1, mIoU, classes_avg_accuracy, global_accuracy, train_loss, val_loss))
