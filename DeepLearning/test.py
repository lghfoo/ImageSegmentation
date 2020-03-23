from dataset.CamVid import CamVid
import validate
import torch

class TestConfig:
    def __init__(self, 
        batch_size = 4,
        data_root = './data/camvid',
        model_path = './model.pth'
    ):
        self.batch_size = batch_size
        self.criterion = validate.CrossEntropyLoss2d()
        self.data_root = data_root
        self.model_path = model_path

def test(net, test_config):
    net.load_state_dict(torch.load(test_config.model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    testset = CamVid(root=test_config.data_root, split='test')
    
    global_accuracy, classes_avg_accuracy, mIoU, test_loss, classes_accuracy, classes_iou = validate.validate(net, testset, test_config.batch_size, device, test_config.criterion)

    print("-------- Test Summary --------")
    print("mIoU: " + str(mIoU))
    print("classes_avg_accuracy: " + str(classes_avg_accuracy))
    print("global_accuracy: " + str(global_accuracy))
    print("test_loss: " + str(test_loss))
    for i in range(net.num_classes):
        print('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, classes_iou[i], classes_accuracy[i], CamVid.classes[i]))

