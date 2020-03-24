import torch
import torch.nn as nn
import numpy as np
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss

def intersection_and_union(output, target, K, ignore_index=255):
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output.long() == target.long()]
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# input: net, dataset, batch_size, device, criterion
# output: global_accuracy, classes_avg_accuracy, mIoU, val_loss, classes_accuracy, classes_iou
def validate(net, valset, batch_size, device, criterion):
    val_dataloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=True, num_workers=0)
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    iter_count = 0
    val_loss = 0
    with torch.no_grad():
        # i = 0
        for i, data in enumerate(val_dataloader):
            inputs = data[0].to(device)
            labels = data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels.squeeze(1).long())
            _, preds = torch.max(outputs, 1)
            intersection, union, target = intersection_and_union(preds, labels, net.num_classes)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            # i += batch_size
            val_loss += loss.item()
            iter_count += 1
            # break
        # while i + batch_size < len(valset):
        #     inputs_and_labels = [valset[i+j] for j in range(batch_size)]
        #     inputs = torch.stack([inputs_and_labels[i][0] for i in range(batch_size)]).to(device)
        #     labels = torch.stack([torch.as_tensor(np.array(inputs_and_labels[i][1])) for i in range(batch_size)]).to(device)
        #     outputs = net(inputs)
        #     loss = criterion(outputs, labels.squeeze(1).long())
        #     _, preds = torch.max(outputs, 1)
        #     intersection, union, target = intersection_and_union(preds, labels, net.num_classes)
        #     intersection_meter.update(intersection)
        #     union_meter.update(union)
        #     target_meter.update(target)
        #     i += batch_size
        #     val_loss += loss.item()
        #     iter_count += 1
        #     # break
    val_loss /= iter_count
    classes_iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    classes_accuracy = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = classes_iou.mean()
    classes_avg_accuracy = classes_accuracy.mean()
    global_acc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    return global_acc, classes_avg_accuracy, mIoU, val_loss, classes_accuracy, classes_iou