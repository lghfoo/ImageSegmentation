import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class FCN16s(nn.Module):
    def __init__(self, num_classes):
        super(FCN16s, self).__init__()
        self.num_classes = num_classes
        # 卷积层
        # 第一层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=100)    
        self.relu1 = nn.ReLU()                                                                                  
        # 第二层                                                                                    
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)             
        self.relu2 = nn.ReLU()                                              
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第三层
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        # 第四层
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第五层
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        # 第六层
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        # 第七层
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第八层
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU()
        # 第九层
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.ReLU()
        # 第十层
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu10 = nn.ReLU()
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool4_predict = nn.Conv2d(in_channels=512, out_channels=self.num_classes, kernel_size=1, stride=1)
        self.deconv1 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=4, stride=2, bias=False)
        # 第十一层
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu11 = nn.ReLU()
        # 第十二层
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu12 = nn.ReLU()
        # 第十三层
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu13 = nn.ReLU()
        self.max_pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        #全连接层
        # 第一层
        self.fc_conv1 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7)
        self.fc_relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        # 第二层
        self.fc_conv2 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)
        self.fc_relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        # 第三层
        self.fc_conv3 = nn.Conv2d(in_channels=4096, out_channels=self.num_classes, kernel_size=1)
        self.deconv2 = nn.ConvTranspose2d(self.num_classes, self.num_classes, kernel_size=32, stride=16, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.conv1.weight.data = vgg16.features[0].weight.data.clone()
        self.conv2.weight.data = vgg16.features[2].weight.data.clone()
        self.conv3.weight.data = vgg16.features[5].weight.data.clone()
        self.conv4.weight.data = vgg16.features[7].weight.data.clone()
        self.conv5.weight.data = vgg16.features[10].weight.data.clone()
        self.conv6.weight.data = vgg16.features[12].weight.data.clone()
        self.conv7.weight.data = vgg16.features[14].weight.data.clone()
        self.conv8.weight.data = vgg16.features[17].weight.data.clone()
        self.conv9.weight.data = vgg16.features[19].weight.data.clone()
        self.conv10.weight.data = vgg16.features[21].weight.data.clone()
        self.conv11.weight.data = vgg16.features[24].weight.data.clone()
        self.conv12.weight.data = vgg16.features[26].weight.data.clone()
        self.conv13.weight.data = vgg16.features[28].weight.data.clone()

        self.conv1.bias.data = vgg16.features[0].bias.data.clone()
        self.conv2.bias.data = vgg16.features[2].bias.data.clone()
        self.conv3.bias.data = vgg16.features[5].bias.data.clone()
        self.conv4.bias.data = vgg16.features[7].bias.data.clone()
        self.conv5.bias.data = vgg16.features[10].bias.data.clone()
        self.conv6.bias.data = vgg16.features[12].bias.data.clone()
        self.conv7.bias.data = vgg16.features[14].bias.data.clone()
        self.conv8.bias.data = vgg16.features[17].bias.data.clone()
        self.conv9.bias.data = vgg16.features[19].bias.data.clone()
        self.conv10.bias.data = vgg16.features[21].bias.data.clone()
        self.conv11.bias.data = vgg16.features[24].bias.data.clone()
        self.conv12.bias.data = vgg16.features[26].bias.data.clone()
        self.conv13.bias.data = vgg16.features[28].bias.data.clone()


        def bilinear_kernel(in_channels, out_channels, kernel_size):
            '''
            return a bilinear filter tensor
            '''
            factor = (kernel_size + 1) // 2
            if kernel_size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:kernel_size, :kernel_size]
            filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
            weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
            weight[range(in_channels), range(out_channels), :, :] = filt
            return torch.from_numpy(weight)
        self.deconv1.weight.data = bilinear_kernel(self.num_classes, self.num_classes, 4)
        self.deconv2.weight.data = bilinear_kernel(self.num_classes, self.num_classes, 32)


    def forward(self, x):
        img_size = x.size()[2:]
        # c1
        x = self.relu1(self.conv1(x))
        # c2
        x = self.max_pool1(self.relu2(self.conv2(x))) # 1/2
        
        # c3
        x = self.relu3(self.conv3(x))
        # c4
        x = self.max_pool2(self.relu4(self.conv4(x))) # 1/4
        
        # c5
        x = self.relu5(self.conv5(x))
        # c6
        x = self.relu6(self.conv6(x))
        # c7
        x = self.max_pool3(self.relu7(self.conv7(x))) # 1/8
        
        # c8
        x = self.relu8(self.conv8(x))
        # c9
        x = self.relu9(self.conv9(x))
        # c10
        x = self.max_pool4(self.relu10(self.conv10(x))) # 1/16
        pool4_predict = self.max_pool4_predict(x)

        # c11
        x = self.relu11(self.conv11(x))
        # c12
        x = self.relu12(self.conv12(x))
        # c13
        x = self.max_pool5(self.relu13(self.conv13(x))) # 1/32

        # fc1
        x = self.dropout1(self.fc_relu1(self.fc_conv1(x)))
        # fc2
        x = self.dropout2(self.fc_relu2(self.fc_conv2(x)))
        # fc3
        pool5_predict = self.fc_conv3(x)

        ######## use Deconv
        def center_crop_tensor(t1, t2):
            (h_t1, w_t1, h_t2, w_t2) = (t1.size()[2], t1.size()[3], t2.size()[2], t2.size()[3])
            return t1[:,:,int((h_t1-h_t2)/2):int((h_t1-h_t2)/2)+h_t2, int((w_t1-w_t2)/2):int((w_t1-w_t2)/2)+w_t2]
        

        upsamled2x = self.deconv2(pool5_predict)
        sigma1 = center_crop_tensor(pool4_predict, upsamled2x) + upsamled2x
        upsamled16x = self.deconv1(sigma1)
        return center_crop_tensor(upsamled16x, torch.Tensor(1, 1, img_size[0], img_size[1]))