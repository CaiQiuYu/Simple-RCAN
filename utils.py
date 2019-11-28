import cv2
import os
import errno
import time
import numpy as np
import torch
import sys
import math
from torch.autograd import Variable


def horizontal_flip(image, axis):
    if axis != 2:
        image = cv2.flip(image, axis)
    return image


def save_checkpoint(model, epoch):
    model_folder = "log/"
    model_out_path = model_folder + "model_epoch_{}_rcan.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(model.module.state_dict(), model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def psnr_cal(pred, gt):
    batch = pred.shape[0]
    psnr = 0
    for i in range(batch):
        for j in range(3):
            pr = pred[i, j, :, :]
            hd = gt[i, j, :, :]

            imdff = pr - hd
            rmse = math.sqrt(np.mean(imdff ** 2))
            if rmse == 0:
                psnr = psnr + 45
                continue
            psnr = psnr + 20 * math.log10(255.0 / rmse)
    return psnr / (batch*3)


def train_content(training_data_loader, optimizer, model, model_feature, criterion_pix, criterion_fea, epoch, opt, iterations):
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_pix = AverageMeter()
    losses_fea = AverageMeter()
    psnrs = AverageMeter()

    model.train()
    end = time.time()
    for iteration, batch in enumerate(training_data_loader):
        data_time.update(time.time() - end)
        data_x, data_y = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        data_x = data_x.type(torch.FloatTensor)
        data_y = data_y.type(torch.FloatTensor)

        data_x = data_x.cuda()
        data_y = data_y.cuda()

        pred = model(data_x)
        # pix loss
        loss_pix = criterion_pix(pred, data_y)
        # content loss
        real_fea = model_feature(data_y).detach()
        fake_fea = model_feature(pred)
        loss_fea = criterion_fea(fake_fea, real_fea)
        # add
        total_loss = loss_pix + loss_fea

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        pred = pred.cpu()
        pred = pred.detach().numpy().astype(np.float32)

        data_y = data_y.cpu()
        data_y = data_y.numpy().astype(np.float32)

        psnr = psnr_cal(pred, data_y)

        mean_loss_pix = loss_pix.item()
        mean_loss_fea = loss_fea.item()
        losses_fea.update(mean_loss_fea)
        losses_pix.update(mean_loss_pix)
        psnrs.update(psnr)

        batch_time.update(time.time() - end)
        end = time.time()
        if iteration % opt.print_freq == 0:
            print('Epoch:[{0}/{1}][{2}/{3}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_pix: {losses_pix.val:.3f} ({losses_pix.avg:.3f})\t'
                  'Loss_fea: {losses_fea.val:.3f} ({losses_fea.avg:.3f})\t'
                  'PNSR: {psnrs.val:.3f} ({psnrs.avg:.3f})'
                  .format(epoch, opt.epochs, iteration, iterations//opt.batch_size,
                          batch_time=batch_time, data_time=data_time, losses_pix=losses_pix,
                          losses_fea=losses_fea, psnrs=psnrs))


def train(training_data_loader, optimizer, model, criterion, epoch, opt, iterations):
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    psnrs = AverageMeter()

    model.train()
    end = time.time()
    for iteration, batch in enumerate(training_data_loader):
        data_time.update(time.time() - end)
        data_x, data_y = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        data_x = data_x.type(torch.FloatTensor)
        data_y = data_y.type(torch.FloatTensor)

        data_x = data_x.cuda()
        data_y = data_y.cuda()

        pred = model(data_x)
        # pix loss
        loss = criterion(pred, data_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = pred.cpu()
        pred = pred.detach().numpy().astype(np.float32)

        data_y = data_y.cpu()
        data_y = data_y.numpy().astype(np.float32)

        psnr = psnr_cal(pred, data_y)

        mean_loss = loss.item() / (opt.batch_size*opt.n_colors*((opt.patch_size*opt.scale)**2))
        losses.update(mean_loss)
        psnrs.update(psnr)

        batch_time.update(time.time() - end)
        end = time.time()
        if iteration % opt.print_freq == 0:
            print('Epoch:[{0}/{1}][{2}/{3}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {losses.val:.3f} ({losses.avg:.3f})\t'
                  'PNSR: {psnrs.val:.3f} ({psnrs.avg:.3f})'
                  .format(epoch, opt.epochs, iteration, iterations//opt.batch_size,
                          batch_time=batch_time, data_time=data_time, losses=losses, psnrs=psnrs))
