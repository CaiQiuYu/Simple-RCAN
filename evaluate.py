import argparse
import os
import cv2
import os.path as osp
import glob
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from tqdm import tqdm
import torch.nn as nn
from model.rcan import RCAN


def main(arg):
    print("====>> Read file list")
    test_img_folder = '/media/ltelab/D/caiqiuyu/data/VideoSR/image/test_lr'
    folder_lists = sorted(os.listdir(test_img_folder))
    device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True
    print("===> Building model")
    model = RCAN(arg)
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    # if arg.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(arg.resume))
    #         checkpoint = torch.load(arg.resume)
    #         new_state_dict = OrderedDict()
    #         for k, v in checkpoint.items():
    #             namekey = k[7:]  # remove `module.`
    #             new_state_dict[namekey] = v
    #         # load params
    #         model.load_state_dict(new_state_dict)
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))
    model.load_state_dict(torch.load(arg.resume), strict=True)

    # model = model.to(device)
    model.eval()
    print('Model path {:s}. \nTesting...'.format(arg.resume))

    for path in folder_lists:
        new_path = test_img_folder + '/' + path + '/*.png'
        image_lists = sorted(glob.glob(new_path))
        if not os.path.exists('results/' + path):
            os.mkdir('results/' + path)
        for image in tqdm(image_lists):
            image_name = image.split('/')[-1]
            # read images
            img = cv2.imread(image, cv2.IMREAD_COLOR)
            img = img * 1.0
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)

            with torch.no_grad():
                output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 255).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = output.round()
            cv2.imwrite('results/{0}/{1}'.format(path, image_name), output)

    # print("===> Setting GPU")
    # model = nn.DataParallel(model, device_ids=device_ids)
    # model = model.cuda()
    #
    # # optionally resume from a checkpoint
    # if arg.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(arg.resume))
    #         checkpoint = torch.load(arg.resume)
    #         new_state_dict = OrderedDict()
    #         for k, v in checkpoint.items():
    #             namekey = 'module.' + k  # remove `module.`
    #             new_state_dict[namekey] = v
    #         # load params
    #         model.load_state_dict(new_state_dict)
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))
    #
    # print("===> Testing")
    # model.eval()
    # image_eval(model, rtc_lr_list)


def image_eval(model, file_list):
    for image_file in file_list:
        print(image_file)
        lr_data = cv2.imread(image_file)
        lr_data = cv2.cvtColor(lr_data, cv2.COLOR_BGR2RGB)
        lr_data = np.array(lr_data)
        lr_data = np.ascontiguousarray(np.transpose(lr_data, (2, 0, 1)))
        lr_data = np.expand_dims(lr_data, 0)
        lr_data = torch.from_numpy(lr_data)
        test_data = lr_data.type(torch.FloatTensor)
        test_data = test_data.to(torch.device("cuda"))
        output = model(test_data)
        output = output.data.float().cpu().squeeze(0).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        output[output > 255] = 255
        output[output < 0] = 0
        output = np.array(output, dtype=np.uint8)
        image_name = image_file.split('/')[9]
        image_path = args.AIRTC_HR + image_name.replace('x4', 'original')
        print(image_path)
        cv2.imwrite(image_path, output, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    working_dir = osp.dirname(osp.abspath(__file__))

    # model parameter
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--patch_size', default=64, type=int)
    parser.add_argument('--batch_size', default=8*2, type=int)
    parser.add_argument('--step_batch_size', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument('--epochs', type=int, default=8000)
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
    parser.add_argument('--HejingTest', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'datasets/AIRTC/TEST/'))

    # check point
    parser.add_argument("--resume", default='log/model_epoch_43_rcan.pth', type=str)
    parser.add_argument('--print_freq', default=1, type=int)
    parser.add_argument("--logs_dir", default='log/', type=str)

    args = parser.parse_args()
    main(args)
