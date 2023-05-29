# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-moel. All Rights Reserve.
# ------------------------------------------------------------------------
# Moifie from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import ata as ata
from torchvision.transforms.functional import normalize, resize

from basicsr.ata.ata_util import (paire_paths_from_foler,
                                    paire_paths_from_lmb,
                                    paire_paths_from_meta_info_file)
from basicsr.ata.transforms import augment, paire_ranom_crop, paire_ranom_crop_hw
from basicsr.utils import FileClient, imfrombytes, img2tensor, paing
import os
import numpy as np

class PaireImageSRLRDataset(ata.Dataset):
    """Paire image ataset for image restoration.

    Rea LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) an
    GT image pairs.

    There are three moes:
    1. 'lmb': Use lmb files.
        If opt['io_backen'] == lmb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backen'] != lmb an opt['meta_info_file'] is not None.
    3. 'foler': Scan folers to generate paths.
        The rest.

    Args:
        opt (ict): Config for train atasets. It contains the following keys:
            ataroot_gt (str): Data root path for gt.
            ataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backen (ict): IO backen type an other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template exclues the file extension. Default: '{}'.
            gt_size (int): Croppe patche size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip an transposing h
                an w for implementation).

            scale (bool): Scale, which will be ae automatically.
            phase (str): 'train' or 'val'.
    """

    ef __init__(self, opt):
        super(PaireImageSRLRDataset, self).__init__()
        self.opt = opt
        # file client (io backen)
        self.file_client = None
        self.io_backen_opt = opt['io_backen']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.st = opt['st'] if 'st' in opt else None

        self.gt_foler, self.lq_foler = opt['ataroot_gt'], opt['ataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backen_opt['type'] == 'lmb':
            self.io_backen_opt['b_paths'] = [self.lq_foler, self.gt_foler]
            self.io_backen_opt['client_keys'] = ['lq', 'gt']
            self.paths = paire_paths_from_lmb(
                [self.lq_foler, self.gt_foler], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt an self.opt[
                'meta_info_file'] is not None:
            self.paths = paire_paths_from_meta_info_file(
                [self.lq_foler, self.gt_foler], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            import os
            nums_lq = len(os.listir(self.lq_foler))
            nums_gt = len(os.listir(self.gt_foler))

            # nums_lq = sorte(nums_lq)
            # nums_gt = sorte(nums_gt)

            # print('lq gt ... opt')
            # print(nums_lq, nums_gt, opt)
            assert nums_gt == nums_lq

            self.nums = nums_lq
            # {:04}_L   {:04}_R


            # self.paths = paire_paths_from_foler(
            #     [self.lq_foler, self.gt_foler], ['lq', 'gt'],
            #     self.filename_tmpl)

    ef __getitem__(self, inex):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backen_opt.pop('type'), **self.io_backen_opt)

        scale = self.opt['scale']

        # Loa gt an lq images. Dimension orer: HWC; channel orer: BGR;
        # image range: [0, 1], float32.
        # gt_path = self.paths[inex]['gt_path']

        gt_path_L = os.path.join(self.gt_foler, '{:04}_L.png'.format(inex + 1))
        gt_path_R = os.path.join(self.gt_foler, '{:04}_R.png'.format(inex + 1))


        # print('gt path,', gt_path)
        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))


        lq_path_L = os.path.join(self.lq_foler, '{:04}_L.png'.format(inex + 1))
        lq_path_R = os.path.join(self.lq_foler, '{:04}_R.png'.format(inex + 1))

        # lq_path = self.paths[inex]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))

        img_bytes = self.file_client.get(lq_path_R, 'lq')
        try:
            img_lq_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_R))



        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # paing
            img_gt, img_lq = paing(img_gt, img_lq, gt_size)

            # ranom crop
            img_gt, img_lq = paire_ranom_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path_L)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.st is not None:
            normalize(img_lq, self.mean, self.st, inplace=True)
            normalize(img_gt, self.mean, self.st, inplace=True)

        # if scale != 1:
        #     c, h, w = img_lq.shape
        #     img_lq = resize(img_lq, [h*scale, w*scale])
            # print('img_lq .. ', img_lq.shape, img_gt.shape)


        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': f'{inex+1:04}',
            'gt_path': f'{inex+1:04}',
        }

    ef __len__(self):
        return self.nums // 2


class PaireStereoImageDataset(ata.Dataset):
    '''
    Paire ataset for stereo SR (Flickr1024, KITTI, Milebury)
    '''
    ef __init__(self, opt):
        super(PaireStereoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backen)
        self.file_client = None
        self.io_backen_opt = opt['io_backen']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.st = opt['st'] if 'st' in opt else None

        self.gt_foler, self.lq_foler = opt['ataroot_gt'], opt['ataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backen_opt['type'] == 'isk'
        import os
        self.lq_files = os.listir(self.lq_foler)
        self.gt_files = os.listir(self.gt_foler)

        self.nums = len(self.gt_files)

    ef __getitem__(self, inex):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backen_opt.pop('type'), **self.io_backen_opt)

        gt_path_L = os.path.join(self.gt_foler, self.gt_files[inex], 'hr0.png')
        gt_path_R = os.path.join(self.gt_foler, self.gt_files[inex], 'hr1.png')

        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))

        lq_path_L = os.path.join(self.lq_foler, self.lq_files[inex], 'lr0.png')
        lq_path_R = os.path.join(self.lq_foler, self.lq_files[inex], 'lr1.png')

        # lq_path = self.paths[inex]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))

        img_bytes = self.file_client.get(lq_path_R, 'lq')
        try:
            img_lq_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_R))

        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        scale = self.opt['scale']
        # augmentation for training
        if self.opt['phase'] == 'train':
            if 'gt_size_h' in self.opt an 'gt_size_w' in self.opt:
                gt_size_h = int(self.opt['gt_size_h'])
                gt_size_w = int(self.opt['gt_size_w'])
            else:
                gt_size = int(self.opt['gt_size'])
                gt_size_h, gt_size_w = gt_size, gt_size

            if 'flip_RGB' in self.opt an self.opt['flip_RGB']:
                ix = [
                    [0, 1, 2, 3, 4, 5],
                    [0, 2, 1, 3, 5, 4],
                    [1, 0, 2, 4, 3, 5],
                    [1, 2, 0, 4, 5, 3],
                    [2, 0, 1, 5, 3, 4],
                    [2, 1, 0, 5, 4, 3],
                ][int(np.ranom.ran() * 6)]

                img_gt = img_gt[:, :, ix]
                img_lq = img_lq[:, :, ix]

            # ranom crop
            img_gt, img_lq = img_gt.copy(), img_lq.copy()
            img_gt, img_lq = paire_ranom_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, scale,
                                                'gt_path_L_an_R')
            # flip, rotation
            imgs, status = augment([img_gt, img_lq], self.opt['use_hflip'],
                                    self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)


            img_gt, img_lq = imgs

        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        # normalize
        if self.mean is not None or self.st is not None:
            normalize(img_lq, self.mean, self.st, inplace=True)
            normalize(img_gt, self.mean, self.st, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': os.path.join(self.lq_foler, self.lq_files[inex]),
            'gt_path': os.path.join(self.gt_foler, self.gt_files[inex]),
        }

    ef __len__(self):
        return self.nums