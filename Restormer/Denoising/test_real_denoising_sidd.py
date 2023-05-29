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
# 140.99 26.11 M

print('mac', macs, params)


# Process SIDD
input_dir = "../../../../../data/SIDD_sRGB/"

filepath = os.path.join(input_dir, 'ValidationNoisyBlocksSrgb.mat')
img = sio.loadmat(filepath)
Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
Inoisy /=255.

filepath = os.path.join(input_dir, 'ValidationGtBlocksSrgb.mat')
img = sio.loadmat(filepath)
GT = np.float32(np.array(img['ValidationGtBlocksSrgb']))
GT /=255.

restored = np.zeros_like(Inoisy)

def padr(img):
    pad = 20
    pad_mod = 'reflect'
    img = F.pad(input=img[:,:,pad:-pad,pad:-pad], pad=(pad,pad,pad,pad), mode=pad_mod)
    return img

device = torch.device('cuda')


psnr_list =[]
ssim_list = []

print("Inoisy:", Inoisy.shape)

with torch.no_grad():
    for i in tqdm(range(Inoisy.shape[0])):  # id

        input_noisy = torch.from_numpy(Inoisy[i]).unsqueeze(0).permute(0,3,1,2).cuda()
        input_GT = torch.from_numpy(GT[i]).unsqueeze(0).permute(0,3,1,2).cuda()

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

print("SIDD PSNR: ", np.mean(psnr_list), ", SSIM: ", np.mean(ssim_list))



# PolyU



# save denoised data
# sio.savemat(os.path.join(result_dir_mat, 'Idenoised.mat'), {"Idenoised": restored,})
