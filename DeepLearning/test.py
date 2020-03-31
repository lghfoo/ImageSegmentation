import validate
import torch
import time
import os
import errno
test_log_file = None
def log(msg):
    global test_log_file
    print(msg)
    test_log_file.write(str(msg) + '\n')

class TestConfig:
    def __init__(self, 
        batch_size = 4,
        data_root = './data/camvid',
        model_path = './model.pth',
        dataset = 'camvid11',
        split = 'test'
    ):
        self.batch_size = batch_size
        self.criterion = validate.CrossEntropyLoss2d()
        self.data_root = data_root
        self.model_path = model_path
        self.dataset = dataset
        self.split = split

def test(net, test_config):
    global test_log_file
    test_filename = './log/test/{}.{}.txt'.format(type(net).__name__, time.strftime("%a_%b_%d_%H_%M_%S_%Y", time.localtime()))
    if not os.path.exists(os.path.dirname(test_filename)):
        try:
            os.makedirs(os.path.dirname(test_filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    test_log_file = open(test_filename, "w")
    log('******** test begin [{}] ********'.format(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())))
    log('batch_size: {}'.format(test_config.batch_size))
    log('model_path: {}'.format(test_config.model_path))
    log('split: {}'.format(test_config.split))
    log('dataset: {}'.format(test_config.dataset))
    
    net.load_state_dict(torch.load(test_config.model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    testset = validate.get_dataset(test_config.dataset, test_config.split, test_config.data_root)
    
    global_accuracy, classes_avg_accuracy, mIoU, test_loss, classes_accuracy, classes_iou = validate.validate(net, testset, test_config.batch_size, device, test_config.criterion)

    log("-------- Test Summary --------")
    log("mIoU: " + str(mIoU))
    log("classes_avg_accuracy: " + str(classes_avg_accuracy))
    log("global_accuracy: " + str(global_accuracy))
    log("test_loss: " + str(test_loss))
    for i in range(net.num_classes):
        log('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, classes_iou[i], classes_accuracy[i], validate.get_dataset_classes(test_config.dataset)[i]))
    
    log('******** test end [{}] ********'.format(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())))
    test_log_file.close()

