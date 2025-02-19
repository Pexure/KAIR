import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    # parser.add_argument('--load', type=str, help='Path ckpt.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    opt['rank'], opt['world_size'] = get_dist_info()
    
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()  # load ckpt

    # -------------------------------
    # 6) testing
    # -------------------------------
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_psnry = 0.0
    avg_ssimy = 0.0
    idx = 0

    for test_data in test_loader:
        idx += 1
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)

        img_dir = os.path.join(opt['path']['images'], img_name)
        util.mkdir(img_dir)

        model.feed_data(test_data)
        model.test()

        visuals = model.current_visuals()
        E_img = util.tensor2uint(visuals['E'])  # E_img: HWC-RGB, uint8[0, 255]
        H_img = util.tensor2uint(visuals['H'])  # H_img: HWC-RGB, uint8[0, 255]

        # -----------------------
        # save estimated image E
        # -----------------------
        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
        util.imsave(E_img, save_img_path)  # has no effect on E_img

        # -----------------------
        # calculate PSNR
        # -----------------------
        E_img = E_img[:, :, [2, 1, 0]]  # HWC-BGR; make a copy, do not share memory
        H_img = H_img[:, :, [2, 1, 0]]  # HWC-BGR;

        current_psnr = util.calculate_psnr(E_img, H_img, border=border)
        current_ssim = util.calculate_ssim(E_img, H_img, border=border)

        E_img = util.bgr2ycbcr(E_img.astype(np.float32) / 255.) * 255.
        H_img = util.bgr2ycbcr(H_img.astype(np.float32) / 255.) * 255.
        current_psnry = util.calculate_psnr(E_img, H_img, border=border)
        current_ssimy = util.calculate_ssim(E_img, H_img, border=border)

        print('PNSR_Y: {:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnry))
        print('SSIM_Y: {:->4d}--> {:>10s} | {:<4.4f}dB'.format(idx, image_name_ext, current_ssimy))

        avg_psnr += current_psnr
        avg_ssim += current_ssim
        avg_psnry += current_psnry
        avg_ssimy += current_ssimy

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_psnry = avg_psnry / idx
    avg_ssimy = avg_ssimy / idx

    # testing log
    print('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(0, current_step, avg_psnr))
    print('<epoch:{:3d}, iter:{:8,d}, Average SSIM : {:<.4f}dB\n'.format(0, current_step, avg_ssim))
    print('<epoch:{:3d}, iter:{:8,d}, Average PSNR_Y : {:<.2f}dB\n'.format(0, current_step, avg_psnry))
    print('<epoch:{:3d}, iter:{:8,d}, Average SSIM_Y : {:<.4f}dB\n'.format(0, current_step, avg_ssimy))


if __name__ == '__main__':
    main()
