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
from glob import glob
from PIL import Image
import random
import numpy as np

from torch.utils.data import Dataset

class PolyU(Dataset):
    def __init__(self,  **kwargs):
        super().__init__( **kwargs)

        self.paths_L = glob("../../../../data/denoising/PolyU/gt/*")
        self.paths_H = glob("../../../../data/denoising/PolyU/noisy/*")
        self.paths_H.sort()
        self.paths_L.sort()

        # *255

    def __len__(self):
        return len(self.paths_H)

    def get_img_by_index(self, index):
        H_path = self.paths_H[index]
        L_path = self.paths_L[index].replace("/gt", "/noisy")

        img_H = Image.open(H_path)
        img_L = Image.open(L_path)

        img_H = np.asarray(img_H).transpose(2, 0, 1)
        img_L = np.asarray(img_L).transpose(2, 0, 1)

        # (npImg_noisy, (2, 0, 1)) / 255)

        if np.max(img_H) > 1.1:
            img_H = img_H / 255
            img_L = img_L / 255

        return img_H, img_L


    def __getitem__(self, idx):
        '''
        final dictionary shape of data:
        {'clean', 'syn_noisy', 'real_noisy', 'noisy (any of real[first priority] and syn)', etc}
        '''
        # calculate data index
        data_idx = idx #% self.n_data

        # load data
        img_H, img_L = self.get_img_by_index(data_idx)


        return np.array(img_H, dtype=np.float32),  np.array(img_L, dtype=np.float32)


D = PolyU()
from torch.utils.data import DataLoader

test_dataloader = DataLoader(D, batch_size=1,
                             shuffle=False, num_workers=8, pin_memory=True)  #




def main():

    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    ## 2. run inference
    opt['dist'] = False
    model = create_model(opt)


    psnr_list = []
    ssim_list = []


    # SIDD
    with torch.no_grad():
        for batch_idx, (input_GT, input_noisy) in enumerate(test_dataloader):  # id

            input_noisy = input_noisy.float().cuda()
            input_GT = input_GT.float().cuda()

            ## 1. read image
            (B, C, W, H) = input_noisy.shape

            output = torch.zeros_like(input_noisy).cuda()

            W_st = W // 256 + 1
            H_st = H // 256 + 1
            pad = 20

            for i in range(W_st):
                for j in range(H_st):
                    # noisy_patch = padr(input_noisy[:, :, i * 256:(i + 1) * 256, j * 256:(j + 1) * 256])
                    noisy_patch = input_noisy[:, :, i * 256:(i + 1) * 256, j * 256:(j + 1) * 256]

                    model.net_g.eval()
                    clean = model.net_g(noisy_patch)

                    output[:, :, i * 256:(i + 1) * 256, j * 256:(j + 1) * 256] = clean


            print('img', output.max(), input_GT.max())

            psnr = compare_psnr(output.cpu().numpy()[0], input_GT.cpu().numpy()[0], data_range=1)
            ssim = compare_ssim(output.cpu().numpy()[0], input_GT.cpu().numpy()[0],
                                data_range=1, multichannel=True,
                                channel_axis=0)

            psnr_list.append(psnr)
            ssim_list.append(ssim)



        print("SIDD PSNR: ", np.mean(psnr_list), ", SSIM: ", np.mean(ssim_list))


if __name__ == '__main__':
    main()

