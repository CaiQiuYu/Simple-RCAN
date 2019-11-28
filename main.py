import argparse
import os
import sys
import os.path as osp
import glob
import numpy as np
import torch
from utils import Logger
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from model.rcan import RCAN
from dataloder import DatasetLoaderWithHR, DatasetLoader
from utils import save_checkpoint, train


def main(arg):
    sys.stdout = Logger(osp.join(args.logs_dir, 'log_1109_rcan.txt'))
    print("====>> Read file list")
    file_name = sorted(os.listdir(arg.VIDEO4K_LR))
    lr_list = []
    hr_list = []
    for fi in file_name:
        lr_tmp = sorted(glob.glob(arg.VIDEO4K_LR + '/' + fi + '/*.png'))
        lr_list.extend(lr_tmp)
        hr_tmp = sorted(glob.glob(arg.VIDEO4K_HR + '/' + fi + '/*.png'))
        if len(hr_tmp) != 100:
            print(fi)
        hr_list.extend(hr_tmp)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print(len(lr_list))
    print(len(hr_list))
    cudnn.benchmark = True

    print("===> Loading datasets")
    data_set = DatasetLoader(lr_list, hr_list, arg.patch_size, arg.scale)
    train_loader = DataLoader(data_set, batch_size=arg.batch_size,
                              num_workers=arg.workers, shuffle=True, pin_memory=True)

    print("===> Building model")
    device_ids = [0]
    model = RCAN(arg)
    criterion = nn.L1Loss(size_average=False)

    print("===> Setting GPU")
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()
    criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if arg.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(arg.resume))
            checkpoint = torch.load(arg.resume)
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                namekey = 'module.' + k  # remove `module.`
                new_state_dict[namekey] = v
            # load params
            model.load_state_dict(new_state_dict)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=arg.lr, weight_decay=arg.weight_decay, betas=(0.9, 0.999), eps=1e-08)

    print("===> Training")
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(optimizer, epoch)
        train(train_loader, optimizer, model, criterion, epoch, arg, len(hr_list), )
        save_checkpoint(model, epoch)


def adjust_lr(opt, epoch):
    scale = 0.1
    print('Current lr {}'.format(args.lr))
    if epoch in [200, 300, 350]:
        args.lr *= 0.1
        print('Change lr to {}'.format(args.lr))
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] * scale


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    working_dir = osp.dirname(osp.abspath(__file__))
    # model parameter
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--patch_size', default=64, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--step_batch_size', default=1, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--start_epoch", default=8, type=int)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument("--n_res_blocks", type=int, default=20)
    parser.add_argument("--n_feats", type=int, default=64)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--res_scale', type=float, default=0.1,
                        help='residual scaling')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--n_resgroups', type=int, default=10,
                        help='number of residual groups')
    parser.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')

    # path
    parser.add_argument('--VIDEO4K_LR', type=str, metavar='PATH',
                        default='/media/ltelab/D/caiqiuyu/data/VideoSR/image/lr')
    parser.add_argument('--VIDEO4K_HR', type=str, metavar='PATH',
                        default='/media/ltelab/D/caiqiuyu/data/VideoSR/image/hr')

    # check point
    parser.add_argument("--resume", default='log/model_epoch_7_rcan.pth', type=str)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument("--logs_dir", default='log/', type=str)

    args = parser.parse_args()
    main(args)
