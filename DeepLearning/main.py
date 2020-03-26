import argparse
from argparse import RawTextHelpFormatter
import torch
from net.fcn.FCNAlex import AlexNetFCN
from net.fcn.FCN8s import FCN8s
from net.fcn.FCN16s import FCN16s
from net.fcn.FCN32s import FCN32s
from dataset.CamVid import CamVid
from dataset.CamVid import CamVid11
import train as trainer
import test as tester
import predict as predictor
import validate

detail_usage = """
train model: -train net_to_train -o saved_model_path -l learning_rate -e epoch_count -b batch_size -d data_root -opt optimizer -i trained_model_path -ds dataset
test model: -test net_to_test -i model_path -b batch_size -d data_root -ds dataset
predict: -predict net_to_predict -i model_path -im input_image -o output_image -ds dataset

net_to_train/test/predict: [
    fcn_alex,
    fcn_8s,
    fcn_16s,
    fcn_32s,
    segnet
]

optimizer: [
    sgd,
    adagrad,
    adadelta,
    adam
]

dataset: [
    camvid,
    camvid11
]

examples:
    python main.py -train fcn_alex -o './fcn_alex.pth' -l 0.01 -e 20 -b 4 -d './data/camvid' -opt sgd -i './trained_fcn_alex.pth' -ds 'camvid11'
    python main.py -train fcn_8s -o './fcn_8s.pth' -l 0.01 -e 80 -b 4 -d './data/camvid' -opt sgd -ds 'camvid11'
    python main.py -test fcn_alex -i './fcn_alex.pth' -b 4 -d './data/camvid' -ds 'camvid11'
    python main.py -predict fcn_alex -i './fcn_alex.pth' -im 'xxxxx.png' -o 'xxxxx_seged.png' -ds 'camvid11'
    python main.py -predict fcn_8s -i './fcn_8s.pth' -im '../../CamVid/images/test/0001TP_008550.png' -ds 'camvid11'
"""

def net_from_type_string(net_type, num_classes):
    if net_type == 'fcn_alex':
        return AlexNetFCN(num_classes)
    elif net_type == 'fcn_8s':
        return FCN8s(num_classes)
    elif net_type == 'fcn_16s':
        return FCN16s(num_classes)
    elif net_type == 'fcn_32s':
        return FCN32s(num_classes)
    elif net_type == 'segnet':
        pass
    print('error: unkown net type')
    return None

def get_num_classes(dataset):
    if dataset is None or dataset == 'camvid11':
        return len(CamVid11.classes)
    if dataset == 'camvid':
        return len(CamVid.classes)
    print('error: unkown dataset')
    return 0

def train(args):
    net = net_from_type_string(args.train, get_num_classes(args.ds))
    config = trainer.TrainConfig()
    if args.o is not None:
        config.model_path = args.o
    if args.l is not None:
        config.learning_rate = args.l
    if args.e is not None:
        config.epoch_count = args.e
    if args.b is not None:
        config.batch_size = args.b
    if args.d is not None:
        config.data_root = args.d
    if args.opt is not None:
        config.optimizer = args.opt
    if args.i is not None:
        config.trained_model_path = args.i
    if args.ds is not None:
        config.dataset = args.ds
    trainer.train(net, config)

def test(args):
    net = net_from_type_string(args.test, get_num_classes(args.ds))
    config = tester.TestConfig()
    if args.i is not None:
        config.model_path = args.i
    if args.b is not None:
        config.batch_size = args.b
    if args.d is not None:
        config.data_root = args.d
    if args.ds is not None:
        config.dataset = args.ds
    tester.test(net, config)

def predict(args):
    assert args.i is not None and args. im is not None and args.ds is not None
    net = net_from_type_string(args.predict, get_num_classes(args.ds))
    net.load_state_dict(torch.load(args.i))
    predictor.predict(net, args.im, args.o, validate.get_dataset_classes(args.ds))

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, epilog=detail_usage)
    parser.add_argument('-train', help='train model')
    parser.add_argument('-test', help='test model')
    parser.add_argument('-predict', help='predict')
    parser.add_argument('-i', help='input model')
    parser.add_argument('-o', help='output model/image')
    parser.add_argument('-l', type=float, help='learning rate')
    parser.add_argument('-e', type=int, help='epoch count')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-im', help='input image')
    parser.add_argument('-d', help='data root')
    parser.add_argument('-opt', help='optimizer')
    parser.add_argument('-ds', help='dataset')
    args = parser.parse_args()
    if args.train is not None:
        train(args)
    elif args.test is not None:
        test(args)
    elif args.predict is not None:
        predict(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()