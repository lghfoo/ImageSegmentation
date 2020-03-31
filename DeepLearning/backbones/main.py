import argparse
from argparse import RawTextHelpFormatter
import torch
import os
from AlexNet import AlexNet
from VGG16 import VGG16
import ResNet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms
dataset = 'cifar10'
num_classes = 10
detail_usage = """
examples:
sudo python main.py -train resnet -o './resnet.pth' -l 0.01 -opt adam -d './data/cifar10' -e 10 -b 4
sudo python main.py -test resnet -i './resnet.pth' -d './data/cifar10' -b 4
"""

def net_from_type_string(net_type, num_classes):
    if net_type == 'alexnet':
        return AlexNet(num_classes)
    elif net_type == 'vgg16':
        return AlexNet(num_classes)
    elif net_type == 'resnet':
        return ResNet.make_resnet(50, num_classes)
    print('error: unkown net type')
    return None

################ train ################
def train(args):
    net = net_from_type_string(args.train, num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    #### config ####
    criterion = nn.CrossEntropyLoss()
    data_root = args.d
    batch_size = args.b
    epoch_count = args.e
    learning_rate = args.l
    output_path = args.o
    print_size = 30 if args.ps is None else args.ps
    optimizer = None
    if args.opt == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=learning_rate)
    elif args.opt == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=learning_rate)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=0.9)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    else:
        print('unkown optimizer')
        return
    #### dataset ####
    transform = transforms.Compose(
        [transforms.Resize(size=net.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ori_trainset = CIFAR10(root=data_root, train=True, download=True, transform=transform)
    train_size = int(len(ori_trainset)*(0.9))
    val_size = len(ori_trainset) - train_size
    #### training ####
    train_info = []
    for epoch in range(epoch_count):
        trainset, valset = torch.utils.data.random_split(ori_trainset, [train_size, val_size])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                    shuffle=True, num_workers=0)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                    shuffle=True, num_workers=0)
        running_loss = 0.0
        train_loss = 0.0
        iter_count = 0
        iter_loss = 0
        # train
        for i, data in enumerate(trainloader, 0):
            iter_count += 1
            inputs, labels = data[0].to(device), data[1].to(device)#data
            optimizer.zero_grad()
            outputs = net(inputs)
            _,predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            iter_loss = loss.item()
            running_loss += iter_loss
            train_loss += loss.item()
            if i % print_size == print_size-1:    
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_size))
                running_loss = 0.0
        train_loss /= iter_count

        # validate
        correct = 0
        total = 0
        val_loss = 0.0
        iter_count = 0
        with torch.no_grad():
            for data in valloader:
                iter_count += 1
                images, labels = data[0].to(device), data[1].to(device)#data
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= iter_count
        cur_acc = correct / total
        if best_acc is None or cur_acc > best_acc:
            print('save model' + output_path)
            torch.save(net.state_dict(), output_path)
            best_acc  = cur_acc
        train_info.append((epoch, cur_acc, learning_rate, train_loss, val_loss))
        print("epoch %02d, acc %.3f, cur_learning_rate %.9f, train_loss %.4f, val_loss %.4f" % (epoch + 1, cur_acc, learning_rate, train_loss, val_loss))

    print('---------------- Summary ----------------')
    for info in train_info:
        (epoch, cur_acc, learning_rate, train_loss, val_loss) = info
        print("epoch %02d, acc %.3f, cur_learning_rate %.9f, train_loss %.4f, val_loss %.4f" % (epoch + 1, cur_acc, learning_rate, train_loss, val_loss))
    print('Finished Training')

################ test ################
def test(args):
    net = net_from_type_string(args.test, num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    #### config ####
    data_root = args.d
    batch_size = args.b
    #### dataset ####
    transform = transforms.Compose(
        [transforms.Resize(size=net.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    correct = 0
    total = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)#data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    print('Accuracy of the network on the 10000 test images: %.1f %%' % (100 * float(correct) / float(total)))
    for i in range(num_classes):
        print('Accuracy of %5s : %.1f %%, correct: %d, total: %d' % (
            classes[i], 100 * float(class_correct[i]) / float(class_total[i]), class_correct[i], class_total[i]))

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, epilog=detail_usage)
    parser.add_argument('-train', help='train model')
    parser.add_argument('-test', help='test model')
    parser.add_argument('-i', help='input model')
    parser.add_argument('-o', help='output model/image')
    parser.add_argument('-l', type=float, help='learning rate')
    parser.add_argument('-e', type=int, help='epoch count')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-d', help='data root')
    parser.add_argument('-opt', help='optimizer')
    parser.add_argument('-ps', type=int, help='print size when training')
    args = parser.parse_args()
    if args.train is not None:
        train(args)
    elif args.test is not None:
        test(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()