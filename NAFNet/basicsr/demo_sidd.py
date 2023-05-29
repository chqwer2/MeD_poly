# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch

# from data import create_dataloader, create_dataset
from models import create_model
from train import parse_options
from utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite

# from utils import (get_env_info, get_root_logger, get_time_str,
#                            make_exp_dirs)
# from utils.options import dict2str


from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

import numpy as np
import os
import scipy.io as sio

# SIDD
input_dir = "../../../../data/SIDD_sRGB/"

filepath = os.path.join(input_dir, 'ValidationNoisyBlocksSrgb.mat')
img = sio.loadmat(filepath)
Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
Inoisy /=255.

filepath = os.path.join(input_dir, 'ValidationGtBlocksSrgb.mat')
img = sio.loadmat(filepath)
GT = np.float32(np.array(img['ValidationGtBlocksSrgb']))
GT /=255.

from tqdm import tqdm


def main():

    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    ## 2. run inference
    opt['dist'] = False
    model = create_model(opt)



    inp_shape = (3, 256, 256)

    # pip install ptflops
    from ptflops import get_model_complexity_info
    FLOPS = 0
    macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=True)

    # params = float(params[:-4])
    # MACs (G) in log scale
    print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9

    print('mac', macs, params)

    psnr_list = []
    ssim_list = []


    # SIDD
    with torch.no_grad():
        for i in tqdm(range(Inoisy.shape[0])):  # id
            for j in tqdm(range(Inoisy.shape[1])):  # id
                input_noisy = torch.from_numpy(Inoisy[i, j]).unsqueeze(0).permute(0, 3, 1, 2).cuda()
                input_GT = torch.from_numpy(GT[i, j]).unsqueeze(0).permute(0, 3, 1, 2).cuda()

                ## 1. read image

                model.feed_data(data={'lq': input_noisy})

                if model.opt['val'].get('grids', False):
                    model.grids()

                model.test()


                if model.opt['val'].get('grids', False):
                    model.grids_inverse()

                visuals = model.get_current_visuals()
                output = tensor2img([visuals['result']])

            psnr = compare_psnr(output.cpu().numpy()[0], input_GT.cpu().numpy()[0], data_range=1)
            ssim = compare_ssim(output.cpu().numpy()[0], input_GT.cpu().numpy()[0], data_range=1, multichannel=True,
                                channel_axis=0)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

            print('PSNR: ', psnr, 'SSIM: ', ssim)

        print("SIDD PSNR: ", np.mean(psnr_list), ", SSIM: ", np.mean(ssim_list))


if __name__ == '__main__':
    main()

