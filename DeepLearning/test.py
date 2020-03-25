import validate
import torch
import time

test_log_file = None
def log(msg):
    global test_log_file
    print(msg)
    test_log_file.write(str(msg) + '\n')

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
    global test_log_file
    test_log_file = open('./test.{}.log.{}.txt'.format(type(net).__name__, time.strftime("%a_%b_%d_%H_%M_%S_%Y", time.localtime())), "a")
    log('******** test begin [{}] ********'.format(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())))
    
    net.load_state_dict(torch.load(test_config.model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    testset = validate.get_dataset(test_config.dataset, 'test', test_config.data_root)
    
    global_accuracy, classes_avg_accuracy, mIoU, test_loss, classes_accuracy, classes_iou = validate.validate(net, testset, test_config.batch_size, device, test_config.criterion)

    log("-------- Test Summary --------")
    log("mIoU: " + str(mIoU))
    log("classes_avg_accuracy: " + str(classes_avg_accuracy))
    log("global_accuracy: " + str(global_accuracy))
    log("test_loss: " + str(test_loss))
    for i in range(net.num_classes):
        log('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, classes_iou[i], classes_accuracy[i], validate.get_dataset_classes(test_config.dataset)))
    
    log('******** test end [{}] ********'.format(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())))
    test_log_file.close()

