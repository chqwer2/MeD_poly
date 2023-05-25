import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
import time 
import os
import scipy.io
from scipy.ndimage.interpolation import rotate
from multiprocessing import Pool


def process_image(train_noisy):
    STD_train = []
    for h in range(3,train_noisy.shape[1]-3):
        for w in range(3,train_noisy.shape[2]-3):
            STD_train.append(np.std((train_noisy[:,h-3:h+3,w-3:w+3,:]/255).reshape([-1,36,3]),1).reshape([-1,1,3]))   
    return np.mean(np.concatenate(STD_train,1),1)

def horizontal_flip(image, rate=0.5):
    image = image[:, ::-1, :]
    return image


def vertical_flip(image, rate=0.5):
    image = image[::-1, :, :]
    return image

def random_rotation(image, angle):
    h, w, _ = image.shape
    image = rotate(image, angle)
    return image


from glob import glob


class PolyU(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.paths_L = glob("../../../../data/denoising/PolyU/gt/*")
        self.paths_H = glob("../../../../data/denoising/PolyU/noisy/*")
        self.paths_H.sort()
        self.paths_L.sort()
        self.unfold = torch.nn.Unfold(kernel_size=256, padding=0, stride=256)
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



        size = 256    # patch size
        stride = 256  # patch stride

        patches = self.unfold(img_L)  #img_L.unfold(1, size, stride).unfold(2, size, stride).unfold(3, size, stride)
        print(patches.shape)



        # print("img_H:", img_H.shape)
        return np.array(img_L, dtype=np.float32), \
            np.array(img_H, dtype=np.float32),  0, idx



class benchmark_data(Dataset):

    def __init__(self, data_dir, task, transform=None):

        self.task = task
        self.data_dir = data_dir
        files_tmp = open(self.data_dir+'Scene_Instances.txt','r').readlines()
        self.Validation_Gt = scipy.io.loadmat(self.data_dir+'ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb']
        self.Validation_Noisy = scipy.io.loadmat(self.data_dir+'ValidationNoisyBlocksSrgb.mat')['ValidationNoisyBlocksSrgb']
        self.Validation_Gt = self.Validation_Gt.reshape([-1,256,256,3])
        self.Validation_Noisy = self.Validation_Noisy.reshape([-1,256,256,3])
        self.data_num = self.Validation_Noisy.shape[0]
        self.files = []
        for i in range(160):
            f = files_tmp[i].split("\n")[0]
            #if f[-1]=='N':
            if i >=0:
               self.files.append(f)   
        self.indices = self._indices_generator()
        self.patch_size = 40
        
       
        

        
        if os.path.exists(self.data_dir+'/'+'std.npy'):
           STD = np.load(self.data_dir+'/'+'std.npy')
        else:
           STD = process_image(self.Validation_Noisy)
           np.save(self.data_dir+'/'+'std.npy',STD)
        self.std = STD
        
              

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):

        def data_loader():
            
            if self.task=="test": 
               Img_noisy = self.Validation_Noisy[index]
               Img_GT = self.Validation_Gt[index]
               Img_noisy = (np.transpose(Img_noisy,(2, 0, 1))/255)  # 3,256,256
               Img_GT = (np.transpose(Img_GT,(2, 0, 1))/255)   
               std = self.std[index]


            if self.task=="train":
               Img_noisy = self.Validation_Noisy[index]
               Img_GT = self.Validation_Gt[index]
               
               # Augmentation
               horizontal = torch.randint(0,2, (1,))
               vertical = torch.randint(0,2, (1,))
               rand_rot = torch.randint(0,4, (1,))
               rot = [0,90,180,270]
               if horizontal ==1:
                  Img_noisy = horizontal_flip(Img_noisy)
                  Img_GT = horizontal_flip(Img_GT)
               if vertical ==1:
                 Img_noisy = vertical_flip(Img_noisy)
                 Img_GT = vertical_flip(Img_GT)        
               Img_noisy = random_rotation(Img_noisy,rot[rand_rot])
               Img_GT = random_rotation(Img_GT,rot[rand_rot])         
                 
               Img_noisy = (np.transpose(Img_noisy,(2, 0, 1))/255)
               Img_GT = (np.transpose(Img_GT,(2, 0, 1))/255)   
               std = self.std[index]
               x_00 = torch.randint(0, Img_noisy.shape[1] - self.patch_size, (1,))
               y_00 = torch.randint(0, Img_noisy.shape[2] - self.patch_size, (1,))
               Img_noisy = Img_noisy[:, x_00[0]:x_00[0] + self.patch_size, y_00[0]:y_00[0] + self.patch_size]
               Img_GT = Img_GT[:, x_00[0]:x_00[0] + self.patch_size, y_00[0]:y_00[0] + self.patch_size]

 


            
            return np.array(Img_noisy, dtype=np.float32), np.array(Img_GT, dtype=np.float32),  np.array(std, dtype=np.float32), index #,Img_train, Img_train_noisy, std_train[0]


        def _timeprint(isprint, name, prevtime):
            if isprint:
                print('loading {} takes {} secs'.format(name, time() - prevtime))
            return time()

        if torch.is_tensor(index):
            index = index.tolist()

        input_noisy, input_GT, std, idx = data_loader()
        target = {
                  'dir_idx': str(idx)
                  }

        return target, input_noisy, input_GT, std 

    def _indices_generator(self):

        return np.arange(self.data_num,dtype=int)
   
        

if __name__ == "__main__":
    time_print = True

    prev = time()
