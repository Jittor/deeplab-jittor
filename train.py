import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from backbone import resnet50, resnet101
from deeplab import DeepLab
from voc import TrainDataset, ValDataset
import numpy as np
from utils import Evaluator
from tensorboardX import SummaryWriter
import os
jt.flags.use_cuda = 1

def poly_lr_scheduler(opt, init_lr, iter, epoch, max_iter, max_epoch):
    new_lr = init_lr * (1 - float(epoch * max_iter + iter) / (max_epoch * max_iter)) ** 0.9
    opt.lr = new_lr

    print ("epoch ={} iteration = {} new_lr = {}".format(epoch, iter, new_lr))


def train(model, train_loader, optimizer, epoch, init_lr, writer):
    model.train()
    max_iter = len(train_loader)

    for idx, (image, target) in enumerate(train_loader):
        poly_lr_scheduler(optimizer, init_lr, idx, epoch, max_iter, 50)
        image = image.float32()
        pred = model(image)
        loss = nn.cross_entropy_loss(pred, target, ignore_index=255) # fix a bug
        writer.add_scalar('train/total_loss_iter', loss.data, idx + max_iter * epoch)
        optimizer.step (loss)
        print ('Training in epoch {} iteration {} loss = {}'.format(epoch, idx, loss.data[0]))
def val (model, val_loader, epoch, evaluator, writer):
    model.eval()
    evaluator.reset()
    for idx, (image, target) in enumerate(val_loader):
        image = image.float32()
        output = model(image)
        pred = output.data
        target = target.data
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)
        print ('Test in epoch {} iteration {}'.format(epoch, idx))
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    best_miou = 0.0
    writer.add_scalar('val/mIoU', mIoU, epoch)
    writer.add_scalar('val/Acc', Acc, epoch)
    writer.add_scalar('val/Acc_class', Acc_class, epoch)
    writer.add_scalar('val/fwIoU', FWIoU, epoch)

    if (mIoU > best_miou):
        best_miou = mIoU
    print ('Testing result of epoch {} miou = {} Acc = {} Acc_class = {} \
                FWIoU = {} Best Miou = {}'.format(epoch, mIoU, Acc, Acc_class, FWIoU, best_miou))

def main():
    model = DeepLab(output_stride=16, num_classes=21)
    train_loader = TrainDataset(data_root='/home/guomenghao/voc_aug/mydata/', split='train', batch_size=4, shuffle=True)
    val_loader = ValDataset(data_root='/home/guomenghao/voc_aug/mydata/', split='val', batch_size=1, shuffle=False)
    learning_rate = 0.005
    momentum = 0.9
    weight_decay = 1e-4
    optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    writer = SummaryWriter(os.path.join('curve', 'train.events.wo_drop'))
    epochs = 50
    evaluator = Evaluator(21)
    for epoch in range (epochs):
        train(model, train_loader, optimizer, epoch, learning_rate, writer)
        val(model, val_loader, epoch, evaluator, writer)



if __name__ == '__main__' :
    main ()
