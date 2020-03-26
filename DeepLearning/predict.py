from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import time
import os
import errno
import shutil
import torch

def predict(net, input_image_path, output_image_path, classes):
    input_without_ext = os.path.splitext(os.path.basename(input_image_path))[0]
    predict_dir = './predict_results/{}/{}/'.format(type(net).__name__, input_without_ext)
    if not os.path.exists(predict_dir):
        try:
            os.makedirs(predict_dir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    if output_image_path is None:
        output_image_path = os.path.join(predict_dir, input_without_ext + '_predicted.png')
    log_file = open(os.path.join(predict_dir, 'log.{}.txt'.format(time.strftime("%a_%b_%d_%H_%M_%S_%Y", time.localtime()))), "a")
    img = Image.open(input_image_path)
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = input_transform(img).unsqueeze(0)
    #### predict ####
    beg = time.time()
    outputs = net(img_tensor)
    end = time.time()
    log_file.write('time elapsed: {:.3f}'.format((end-beg)*1000))
    _, pred = torch.max(outputs, 1)
    gray_result = pred.squeeze(0)
    
    r_channel = gray_result.clone()
    g_channel = gray_result.clone()
    b_channel = gray_result.clone()
    for category in classes:
        r_channel[r_channel==category.id] = category.color[0]
        g_channel[g_channel==category.id] = category.color[1]
        b_channel[b_channel==category.id] = category.color[2]

    rgb_array = np.zeros((gray_result.size()[0], gray_result.size()[1], 3), 'uint8')
    rgb_array[..., 0] = np.uint8(r_channel)
    rgb_array[..., 1] = np.uint8(g_channel)
    rgb_array[..., 2] = np.uint8(b_channel)
    #### convert result to img ####
    output_img = Image.fromarray(rgb_array)
    #### save result ####
    output_img.save(output_image_path)
    bak_input_path = os.path.join(predict_dir, input_without_ext + '.png')
    if not os.path.exists(bak_input_path):
        shutil.copy(input_image_path, bak_input_path)
    log_file.close()
    print('finished predicting {}'.format(os.path.basename(input_image_path)))