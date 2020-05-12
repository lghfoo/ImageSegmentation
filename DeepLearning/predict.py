from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import time
import os
import errno
import shutil
import torch

def mkdir_if_not_exists(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def get_module(net):
    if hasattr(net, 'module'):
        return net.module
    return net

def predict(net, input_image_path, output_image_path, classes, need_dbl=False):
    images_dir = "./predict_results/images/"
    labels_dir = "./predict_results/labels/"    
    input_without_ext = os.path.splitext(os.path.basename(input_image_path))[0]
    predict_dir = './predict_results/{}/{}/'.format(type(get_module(net)).__name__, input_without_ext)
    mkdir_if_not_exists(predict_dir)
    mkdir_if_not_exists(images_dir)
    mkdir_if_not_exists(labels_dir)
    output_image_path = os.path.join(predict_dir, input_without_ext + '_predicted.png')
    log_file = open(os.path.join(predict_dir, 'log.{}.txt'.format(time.strftime("%a_%b_%d_%H_%M_%S_%Y", time.localtime()))), "w")
    img = Image.open(input_image_path)
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = input_transform(img).unsqueeze(0)
    if need_dbl:
        img_tensor = torch.cat( (img_tensor, img_tensor), 0)  # double the batch size for BatchNormal2d
    #### predict ####
    beg = time.time()
    outputs = net(img_tensor)
    end = time.time()
    log_file.write('input: {}\ntime elapsed: {:.3f} ms\n\n'.format(input_image_path, (end-beg)*1000))
    _, pred = torch.max(outputs, 1)
    if need_dbl:
        pred = pred[0:1, :, :]
    gray_result = pred.squeeze(0)
    r_channel = gray_result.clone()
    g_channel = gray_result.clone()
    b_channel = gray_result.clone()
    for category in classes:
        r_channel[r_channel==category.id] = category.color[0]
        g_channel[g_channel==category.id] = category.color[1]
        b_channel[b_channel==category.id] = category.color[2]

    rgb_array = np.zeros((gray_result.size()[0], gray_result.size()[1], 3), 'uint8')
    rgb_array[..., 0] = np.uint8(r_channel.cpu())
    rgb_array[..., 1] = np.uint8(g_channel.cpu())
    rgb_array[..., 2] = np.uint8(b_channel.cpu())
    #### convert result to img ####
    output_img = Image.fromarray(rgb_array)
    #### save result ####
    output_img.save(output_image_path)
    bak_input_path = os.path.join(images_dir, input_without_ext + '.png')
    bak_labels_path = os.path.join(labels_dir, input_without_ext + '.png')
    if not os.path.exists(bak_input_path):
        shutil.copy(input_image_path, bak_input_path)
    if not os.path.exists(bak_labels_path):
        shutil.copy(input_image_path.replace('images', 'labels'), bak_labels_path)
    log_file.close()
    print('finished predicting {}, use {} ms'.format(os.path.basename(input_image_path), (end-beg)*1000))