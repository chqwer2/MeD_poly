## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


parser = argparse.ArgumentParser(description='Real Image Denoising using Restormer')

parser.add_argument('--input_dir', default='./Datasets/test/SIDD/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Real_Denoising/SIDD/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/real_denoising.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'Options/RealDenoising_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

result_dir_mat = os.path.join(args.result_dir, 'mat')
os.makedirs(result_dir_mat, exist_ok=True)

if args.save_images:
    result_dir_png = os.path.join(args.result_dir, 'png')
    os.makedirs(result_dir_png, exist_ok=True)

model_restoration = Restormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


# MAC
inp_shape = (3, 256, 256)
# pip install ptflops
from ptflops import get_model_complexity_info
FLOPS = 0
macs, params = get_model_complexity_info(model_restoration, inp_shape, verbose=False, print_per_layer_stat=True)

# params = float(params[:-4])
# MACs (G) in log scale
print(params)
macs = float(macs[:-4]) + FLOPS / 10 ** 9

print('mac', macs, params)


# Process SIDD
from torch.utils.data import Dataset

from glob import glob
from PIL import Image
import random
import numpy as np


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

        # patches = self.unfold(img_L)  #img_L.unfold(1, size, stride).unfold(2, size, stride).unfold(3, size, stride)
        # print(patches.shape)



        # print("img_H:", img_H.shape)
        return 0, \
            np.array(img_L, dtype=np.float32),  np.array(img_H, dtype=np.float32), idx


D = PolyU()
from torch.utils.data import DataLoader

test_dataloader = DataLoader(D, batch_size=1,
                             shuffle=False, num_workers=8, pin_memory=True)  #


def padr(img):
    pad = 20
    pad_mod = 'reflect'
    img = F.pad(input=img[:,:,pad:-pad,pad:-pad], pad=(pad,pad,pad,pad), mode=pad_mod)
    return img

device = torch.device('cuda')


psnr_list =[]
ssim_list = []


with torch.no_grad():
    for batch_idx, (input_GT, input_noisy) in enumerate(test_dataloader):  # id

        input_noisy = input_noisy.cuda()/255.
        input_GT = input_GT.cuda()/ 255.

        unfold = torch.nn.Unfold(kernel_size=256, padding=2, stride=256)
        (B, C, W, H) = input_noisy.shape

        output = torch.zeros_like(input_noisy).to(device)
        W_st = W // 256 + 1
        H_st = H // 256 + 1
        pad = 20


        for i in range(W_st):
            for j in range(H_st):

                noisy_patch = padr(input_noisy[:, :, i * 256:(i + 1) * 256, j * 256:(j + 1) * 256])
                clean = model_restoration(noisy_patch)

                output[:, :, i * 256:(i + 1) * 256, j * 256:(j + 1) * 256] = \
                    clean[:, :, pad:-pad, pad:-pad]

                psnr = compare_psnr(output.cpu().numpy(), input_GT.cpu().numpy(), data_range=1)
                ssim = compare_ssim(output.cpu().numpy(), input_GT.cpu().numpy(), data_range=1, multichannel=True,
                                    channel_axis=-1)

                psnr_list.append(psnr)
                ssim_list.append(ssim)



        print('PSNR: ', psnr, 'SSIM: ', ssim)

print("AVG PolyU PSNR: ", np.mean(psnr_list), ", SSIM: ", np.mean(ssim_list))



# PolyU



# save denoised data
# sio.savemat(os.path.join(result_dir_mat, 'Idenoised.mat'), {"Idenoised": restored,})
