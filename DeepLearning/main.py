import argparse
from argparse import RawTextHelpFormatter
import torch
from net.fcn.FCNAlex import AlexNetFCN
from net.fcn.FCN8s import FCN8s
from net.fcn.FCN16s import FCN16s
from net.fcn.FCN32s import FCN32s
from net.fcn.FCNResNet8s import FCNResNet8s
from net.fcn.FCNResNet50 import FCNResNet50
from net.segnet.SegNet import SegNet
from net.pspnet.PSPNet import PSPNet
from net.pfnet.PFNet import PFNet
from net.danet.DANet import DANet
from net.deeplabv3.DeepLabv3 import DeepLabv3
from net.mdanet.MDANet import MDANet
from net.deeplabv3.CustomDeepLabv3 import CustomDeepLabv3
from net.multimodulenet.MMNet import MMNet
from net.emanet.EMANet import EMANet
from dataset.CamVid import CamVid
from dataset.CamVid import CamVid11
from dataset.VOC2012 import VOC2012
from dataset.SiftFlow import SiftFlow
import train as trainer
import test as tester
import predict as predictor
import validate
import os
import subprocess

detail_usage = """
train model: -train net_to_train -o saved_model_path -l learning_rate -e epoch_count -b batch_size -d data_root -opt optimizer -i trained_model_path -ds dataset -sp train
test model: -test net_to_test -i model_path -b batch_size -d data_root -ds dataset -sp test
predict: [-predict net_to_predict -i model_path]/[-predictf nets_file] [-o output_image] -ds dataset [-iml input_images_list_file]/[-im input_image]

net_to_train/test/predict: [
    fcn_alex,
    fcn_8s,
    fcn_16s,
    fcn_32s,
    fcn_resnet,
    fcn_resnet50,
    segnet,
    pspnet,
    danet,
    deeplabv3,
    deeplabv3_res101,
    emanet,
    emanet_res101,
    mmnet,
    mdanet,
]

optimizer: [
    sgd,
    poly,
    adagrad,
    adadelta,
    adam
]

dataset: [
    camvid,
    camvid11,
    voc2012,
    sift_flow,
]

split: [
    train,
    test,
    val
]

examples:
    python main.py -train fcn_alex -o './fcn_alex.pth' -l 0.01 -e 20 -b 4 -d './data/camvid' -opt sgd -i './trained_fcn_alex.pth' -ds 'camvid11'
    python main.py -train fcn_8s -o './fcn_8s.pth' -l 0.01 -e 80 -b 4 -d './data/camvid' -opt sgd -ds 'camvid11'
    python main.py -train emanet_res101 -o './emanet_voc2012.pth' -l 0.01 -e 20 -b 4 -d '../../VOC2012AUG/' -opt adam -ds 'voc2012' -sp train_aug
    python main.py -test fcn_alex -i './fcn_alex.pth' -b 4 -d './data/camvid' -ds 'camvid11'
    python main.py -predict fcn_alex -i './fcn_alex.pth' -im 'xxxxx.png' -o 'xxxxx_seged.png' -ds 'camvid11'
    python main.py -predict fcn_8s -i './fcn_8s.pth' -im '../../CamVid/images/test/0001TP_008550.png' -ds 'camvid11'
    python main.py -predict fcn_8s -i './fcn_8s.pth' -iml './images_to_predict.txt' -ds 'camvid11'
    sudo python main.py -train segnet -o './segnet.pth' -l 0.01 -e 10 -b 4 -d '../../CamVid' -opt adam -ds 'camvid11'
    sudo python main.py -test segnet -i './segnet.pth' -b 4 -d '../../CamVid' -ds 'camvid11' -sp train
    sudo python main.py -predictf './nets_file.txt' -ds 'camvid11' -iml './images_to_predict.txt'
    sudo python ./main.py -train pspnet -o './pspnet.pth' -l 0.01 -e 20 -b 4 -d '../../CamVid' -opt adam -ds 'camvid11'
    python main.py -train deeplabv3_res101 -o './deeplabv3_voc2012.pth' -l 0.01 -e 40 -b 8 -d '../../VOC2012AUG/' -opt adam -ds 'voc2012' -sp train_aug
    python main.py -train deeplabv3_res101 -o './deeplabv3_res101.pth' -l 0.01 -e 40 -b 8 -d '../../CamVid/' -opt adam -ds 'camvid11' -gpus 0
    python main.py -train fcn_16s -o './fcn_16s_sift_flow.pth' -l 0.0001 -e 40 -b 4 -d '../../sift-flow' -opt adam -ds 'sift_flow' -gpus 0
    sudo python ./main.py -train mmnet -o './mmnet.pth' -l 0.0001 -e 80 -b 4 -d '../../CamVid' -opt adam -ds 'camvid11' -gpus 0
    sudo python ./main.py -train fcn_8s -o './fcn_8s.pth' -l 0.0001 -e 80 -b 4 -d '../../CamVid' -opt adam -ds 'camvid11' -gpus 0
    sudo python ./main.py -train mdanet -o './mdanet.pth' -l 0.0001 -e 80 -b 4 -d '../../CamVid' -opt adam -ds 'camvid11' -gpus 0


[Content in images_to_predict.txt]
../../CamVid/images/test/0001TP_008551.png
../../CamVid/images/test/0001TP_008552.png
../../CamVid/images/test/0001TP_008553.png
....
[Content in nets_file.txt]
fcn_8s
./fcn_8s.pth

fcn16s
./fcn_16s.pth

fcn32s
./fcn_32s.pth

fcn_alex
./fcn_alex.pth

segnet
./segnet.pth
"""

def net_from_type_string(net_type, num_classes, output_stride=16):
    if net_type == 'fcn_alex':
        return AlexNetFCN(num_classes)
    elif net_type == 'fcn_8s':
        return FCN8s(num_classes)
    elif net_type == 'fcn_16s':
        return FCN16s(num_classes)
    elif net_type == 'fcn_32s':
        return FCN32s(num_classes)
    elif net_type == 'segnet':
        return SegNet(num_classes)
    elif net_type == 'pspnet':
        return PSPNet(num_classes)
    elif net_type == 'danet':
        return DANet(num_classes)
    elif net_type == 'pfnet':
        return PFNet(num_classes)
    elif net_type == 'fcn_resnet':
        return FCNResNet8s(num_classes)
    elif net_type == 'fcn_resnet50':
        return FCNResNet50(num_classes)
    elif net_type == 'deeplabv3':
        # return DeepLabv3(num_classes)
        return CustomDeepLabv3(num_classes, output_stride=output_stride)
    elif net_type == 'deeplabv3_res101':
        return DeepLabv3(num_classes, 101)
    elif net_type == 'emanet':
        return EMANet(num_classes)
    elif net_type == 'emanet_res101':
        return EMANet(num_classes, 101)
    elif net_type == 'mmnet':
        return MMNet(num_classes)
    elif net_type == 'mdanet':
        return MDANet(num_classes)
    print('error: unkown net type')
    return None

def get_num_classes(dataset):
    if dataset is None or dataset == 'camvid11':
        return len(CamVid11.classes)
    if dataset == 'camvid':
        return len(CamVid.classes)
    if dataset == 'voc2012':
        return len(VOC2012.classes)
    if dataset == 'sift_flow':
        return len(SiftFlow.classes)
    print('error: unkown dataset')
    return 0

def train(args):
    output_stride = 16 if args.ost is None else args.ost
    net = net_from_type_string(args.train, get_num_classes(args.ds), output_stride=output_stride)
    config = trainer.TrainConfig()
    config.num_classes = get_num_classes(args.ds)
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
    if args.sp is not None:
        config.split = args.sp
    if args.gpu is not None:
        config.gpu = args.gpu
    if args.gpus is not None:
        config.gpus = [int(g.strip()) for g in args.gpus.split(',')]
    config.shuffle = args.nsf
    trainer.train(net, config)

def test(args):
    output_stride = 16 if args.ost is None else args.ost
    net = net_from_type_string(args.test, get_num_classes(args.ds), output_stride=output_stride)
    config = tester.TestConfig()
    config.num_classes = get_num_classes(args.ds)
    if args.i is not None:
        config.model_path = args.i
    if args.b is not None:
        config.batch_size = args.b
    if args.d is not None:
        config.data_root = args.d
    if args.ds is not None:
        config.dataset = args.ds
    if args.sp is not None:
        config.split = args.sp
    if args.gpus is not None:
        config.gpus = [int(g.strip()) for g in args.gpus.split(',')]
    net = torch.nn.DataParallel(net, device_ids=config.gpus).cuda()
    tester.test(net, config)

def predict(args):
    def get_gpu_mem_usage():
        p = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE)
        out = p.stdout.read()
        out = str(out, encoding = "utf-8")  
        lines = out.splitlines()
        # for index,line in enumerate(lines):
        #     print(index, line)
        mem_usage = lines[8]
        vals = mem_usage.split(' ')
        # for index, val in enumerate(vals):
        #     print(index, val)
        for val in vals:
            if val.endswith('MiB'):
                return val
        return 'unkown MiB'

    assert (args.i is not None or args.predictf is not None) and (args.im is not None or args.iml is not None) and args.ds is not None
    output_stride = 16 if args.ost is None else args.ost
    use_gpus = [0]
    if args.gpus is not None:
        use_gpus =  [int(g.strip()) for g in args.gpus.split(',')]
    nets = []
    if args.predictf is not None:
        nets_file = open(args.predictf, "r")
        infos = nets_file.read().split('\n\n')
        for info in infos:
            # t = info.split('\n')[0]
            # m = info.split('\n')[1]
            t,m,p = tuple(info.split('\n'))
            nets.append( (t, m, p) )
        nets_file.close()
    else:
        nets.append( (args.predict, args.i, args.para) )
    
    for net_info in nets:
        net_type = net_info[0]
        dbl = net_type in ['pspnet', 'danet', 'deeplabv3']
        net_model = net_info[1]
        parallel = net_info[2]
        if not os.path.exists(net_model):
            print('warning: cannot find {}, skip.'.format(net_model))
            continue
        net = net_from_type_string(net_type, get_num_classes(args.ds), output_stride=output_stride)

        if parallel == 'p':
            net = torch.nn.DataParallel(net, device_ids=use_gpus).cuda()
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            net.to(device)

        net.load_state_dict(torch.load(net_model))
        if args.iml is None:
            predictor.predict(net, args.im, args.o, validate.get_dataset_classes(args.ds), need_dbl=dbl)
        elif os.path.exists(args.iml):
            list_file = open(args.iml, "r")
            lines = list_file.read().split('\n')
            print("#### predict use {} ####".format(type(predictor.get_module(net)).__name__))
            for line in lines:
                if len(line) == 0:
                    continue
                predictor.predict(net, line.strip(), args.o, validate.get_dataset_classes(args.ds), need_dbl=dbl)
            list_file.close()
    mem_usage = get_gpu_mem_usage()
    print('use {}'.format(mem_usage))

def stat_param_cnt():
    targets = [
        'fcn_8s',
        'segnet',
        'pspnet',
        'deeplabv3',
        'danet'
    ]
    for t in targets:
        net = net_from_type_string(t, 12)
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('{}: {:.2f} MB'.format(t, num_params / 1e6))

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, epilog=detail_usage)
    parser.add_argument('-train', help='train model')
    parser.add_argument('-test', help='test model')
    parser.add_argument('-predict', help='predict')
    parser.add_argument('-predictf', help='predict by file')
    parser.add_argument('-i', help='input model')
    parser.add_argument('-o', help='output model/image')
    parser.add_argument('-l', type=float, help='learning rate')
    parser.add_argument('-e', type=int, help='epoch count')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-im', help='input image to predict')
    parser.add_argument('-d', help='data root')
    parser.add_argument('-opt', help='optimizer')
    parser.add_argument('-ds', help='dataset')
    parser.add_argument('-iml', help='input images list file to predict')
    parser.add_argument('-sp', help='the split to test')
    parser.add_argument('-gpu', help='the gpu to use')
    parser.add_argument('-gpus', help='the gpus to use')
    parser.add_argument('-nsf', action='store_false', help='not shuffle training data')
    parser.add_argument('-ost', type=int, help='output stride')
    parser.add_argument('-stat', action='store_true', help='display param count')
    parser.add_argument('-para',help='is model parallel')
    args = parser.parse_args()
    if args.stat:
        stat_param_cnt()
    elif args.train is not None:
        train(args)
    elif args.test is not None:
        test(args)
    elif args.predict is not None or args.predictf is not None:
        predict(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()