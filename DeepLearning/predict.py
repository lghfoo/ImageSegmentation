from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch

def predict(net, input_image_path, output_image_path):
    if output_image_path is None:
        output_image_path = input_image_path.replace('.png', '_predicted.png')
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