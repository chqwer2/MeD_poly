## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import utils
from pdb import set_trace as stx

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

parser = argparse.ArgumentParser(description='Gaussian Color Denoising using Restormer')

parser.add_argument('--input_dir', default='./Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Gaussian_Color_Denoising/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/gaussian_color_denoising', type=str, help='Path to weights')
parser.add_argument('--model_type', required=True, choices=['non_blind','blind'], type=str, help='blind: single model to handle various noise levels. non_blind: separate model for each noise level.')
parser.add_argument('--sigmas', default='15,25,50', type=str, help='Sigma values')

args = parser.parse_args()

####### Load yaml #######
if args.model_type == 'blind':
    yaml_file = 'Options/GaussianColorDenoising_Restormer.yml'
else:
    yaml_file = f'Options/GaussianColorDenoising_RestormerSigma{args.sigmas}.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

sigmas = np.int_(args.sigmas.split(','))

factor = 8

datasets = ['CBSD68']

for sigma_test in sigmas:
    print("Compute results for noise level",sigma_test)
    model_restoration = Restormer(**x['network_g'])
    if args.model_type == 'blind':
        weights = args.weights+'_blind.pth'
    else:
        weights = args.weights + '_sigma' + str(sigma_test) +'.pth'
    checkpoint = torch.load(weights)
    model_restoration.load_state_dict(checkpoint['params'])

    print("===>Testing using weights: ",weights)
    print("------------------------------------------------")
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    for dataset in datasets:
        inp_dir = os.path.join(args.input_dir, dataset)
        inp_dir = "../../../../../data/denoising/CBSD68-dataset/CBSD68/original_png"

        files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.tif')))
        result_dir_tmp = os.path.join(args.result_dir, args.model_type, dataset, str(sigma_test))
        os.makedirs(result_dir_tmp, exist_ok=True)

        psnr = []
        ssim = []

        with torch.no_grad():
            for file_ in tqdm(files):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                img_in = np.float32(utils.load_img(file_))/255.

                np.random.seed(seed=0)  # for reproducibility
                img = img_in.copy()

                img += np.random.normal(0, sigma_test/255., img_in.shape)

                img = torch.from_numpy(img).permute(2,0,1)
                input_ = img.unsqueeze(0).cuda()

                # Padding in case images are not multiples of 8
                h,w = input_.shape[2], input_.shape[3]
                H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
                padh = H-h if h%factor!=0 else 0
                padw = W-w if w%factor!=0 else 0
                input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

                restored = model_restoration(input_)

                # Unpad images to original dimensions
                restored = restored[:,:,:h,:w]

                restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()


                temp_psnr = compare_psnr(restored.cpu().numpy().float(), img_in.cpu().numpy().float(), data_range=1)
                temp_ssim = compare_ssim(restored.cpu().numpy().float(), img_in.cpu().numpy().float(), data_range=1, multichannel=True)
                psnr.append(temp_psnr)
                ssim.append(temp_ssim)


                # save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1])
                # utils.save_img(save_file, img_as_ubyte(restored))
        print("[Sigma: ",sigma_test,"] psnr:", np.mean(psnr), "ssim:", np.mean(ssim))