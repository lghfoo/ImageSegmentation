from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import time
import os
import errno

def predict(net, input_image_path, output_image_path):
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

    img = Image.open(input_image_path)
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = input_transform(img).unsqueeze(0)

    outputs = net(img_tensor)

    _, pred = torch.max(outputs, 1)
    result = pred.squeeze(0).numpy()
    output_img = Image.fromarray(np.uint8(result))
    output_img.save(output_image_path)