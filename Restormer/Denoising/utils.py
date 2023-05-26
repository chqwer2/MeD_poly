## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import cv2
import math



rng = np.random.default_rng(seed=36)

import torch

def _bernoulli(p, shape):
    return torch.rand(shape) <= p


def gaussian(img, sigma):
    # + np.random.normal(0, self.sigma / 255.0, img_L2.shape)
    return img + torch.FloatTensor(img.shape).normal_(mean=0, std=sigma)


def gaussian_localvar(img):
    # var ** 0.5
    noise = np.random.normal(0, torch.sqrt(img * 255).numpy(), img.shape) / 255
    # print("noise max:", np.max(noise))
    # print("noise img:", torch.max(img)
    return img + noise  # torch.FloatTensor(img.shape).normal_(mean=0, std=torch.sqrt(img).numpy())


def salt_pepper(img, amount=0.05, salt_vs_pepper=0.5):
    flipped = _bernoulli(amount, img.shape)
    salted = _bernoulli(salt_vs_pepper, img.shape)
    peppered = ~salted

    img[flipped & salted] = 1
    img[flipped & peppered] = 0

    return img


def salt_pepper_3(img, amount=0.05, salt_vs_pepper=0.3):
    #
    flipped = _bernoulli(amount, img.shape[:2]).unsqueeze(-1).repeat(1, 1, 3)
    salted = _bernoulli(salt_vs_pepper, img.shape[:2]).unsqueeze(-1).repeat(1, 1, 3)
    # flipped = torch.repeat
    peppered = ~salted

    img[flipped & salted] = 1
    img[flipped & peppered] = 0

    return img, flipped & salted


def salt_pepper_3_torch(img, amount=0.05, salt_vs_pepper=0.3):
    peppered = _bernoulli(amount, img.shape[-2:]).unsqueeze(0).repeat(3, 1, 1)

    # salted = _bernoulli(salt_vs_pepper, img.shape[-2:]).unsqueeze(0).repeat(3, 1, 1)
    #
    # # flipped = torch.repeat
    # peppered = ~salted
    #
    # img[flipped & salted] = 1
    img[peppered] = 0

    return img  # , peppered


def poisson(img):
    img_ = (img.opy() * 255).to(torch.uint8)  # .uint8()
    # img = (scipy.misc.imread(filename)).astype(float)
    # noise_mask = numpy.random.poisson(img)

    vals = torch.unique(img_).shape[0]  # length
    vals = 2 ** torch.ceil(torch.as_tensor(np.log2(vals)))  # 255

    img = torch.poisson(img_ * vals) / float(vals) / 255

    return img


def speckle(img, sigma):
    return img + img * torch.FloatTensor(img.shape).normal_(mean=0, std=sigma)



def add_noise(clean, ntype, sigma=None):
    # assert ntype.lower() in ['gaussian', 'gaussian_gray', 'impulse', 'binomial', 'pattern1', 'pattern2', 'pattern3', 'line']

    img = torch.from_numpy(clean.copy())

    if 'gaussian' in ntype:
        noisy = clean + np.random.normal(0, sigma, clean.shape)
        return np.float32(noisy)

    elif  ntype == "poisson":
        noisy = poisson(img)

    elif  ntype == "local_val":
        noisy = gaussian_localvar(img)

    elif  ntype == "s&p":
        noisy = salt_pepper(img, amount=0.05, salt_vs_pepper=0.5)


    elif  ntype == "speckle":
        noisy = speckle(img, sigma)


    elif ntype == 'binomial':
        h, w, c = clean.shape
        mask = np.random.binomial(n=1, p=(1 - sigma), size=(h, w, 1))
        noisy = clean * mask

    elif ntype == 'impulse':
        mask = np.random.binomial(n=1, p=(1 - sigma), size=clean.shape)
        noisy = clean * mask

    elif ntype[:4] == 'line':
        # sigma = 25 / 255.0
        h, w, c = clean.shape
        line_noise = np.ones_like(clean) * np.random.normal(0, sigma, (h, 1, 1))
        noisy = clean + line_noise

    elif ntype[:7] == 'pattern':
        # sigma = 5 / 255.0
        h, w, c = clean.shape
        n_type = int(ntype[7:])

        one_image_noise, _, _ = get_experiment_noise('g%d' % n_type, sigma, 0, (h, w, 3))
        noisy = clean + one_image_noise
    else:
        assert 'not support %s' % args.ntype

    return noisy.numpy()





def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)
