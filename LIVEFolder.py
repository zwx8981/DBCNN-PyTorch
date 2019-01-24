import torch.utils.data as data

from PIL import Image

import os
import os.path
#import math
import scipy.io
import numpy as np
import random


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename

def getDistortionTypeFileName(path, num):
    filename = []
    index = 1
    for i in range(0,num):
        name = '%s%s%s' % ('img',str(index),'.bmp')
        filename.append(os.path.join(path,name))
        index = index + 1
    return filename
        


class LIVEFolder(data.Dataset):

    def __init__(self, root, loader, index, transform=None, target_transform=None):

        self.root = root
        self.loader = loader
        
        self.refpath = os.path.join(self.root, 'refimgs')
        self.refname = getFileName( self.refpath,'.bmp')
        
        self.jp2kroot = os.path.join(self.root, 'jp2k')
        self.jp2kname = getDistortionTypeFileName(self.jp2kroot,227)
        
        self.jpegroot = os.path.join(self.root, 'jpeg')
        self.jpegname = getDistortionTypeFileName(self.jpegroot,233)
        
        self.wnroot = os.path.join(self.root, 'wn')
        self.wnname = getDistortionTypeFileName(self.wnroot,174)
        
        self.gblurroot = os.path.join(self.root, 'gblur')
        self.gblurname = getDistortionTypeFileName(self.gblurroot,174)

        self.fastfadingroot = os.path.join(self.root, 'fastfading')
        self.fastfadingname = getDistortionTypeFileName(self.fastfadingroot,174)
              
        self.imgpath = self.jp2kname + self.jpegname + self.wnname + self.gblurname + self.fastfadingname
        
        self.dmos = scipy.io.loadmat(os.path.join(self.root, 'dmos_realigned.mat'))
        self.labels = self.dmos['dmos_new'].astype(np.float32)   
        #self.labels = self.labels.tolist()[0]
        self.orgs = self.dmos['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(self.root, 'refnames_all.mat'))
        self.refnames_all = refnames_all['refnames_all']
        

        sample = []
        
        for i in range(0, len(index)):
            train_sel = (self.refname[index[i]] == self.refnames_all)
            train_sel = train_sel * ~self.orgs.astype(np.bool_)
            train_sel1 = np.where(train_sel == True)
            train_sel = train_sel1[1].tolist()
            for j, item in enumerate(train_sel):
                sample.append((self.imgpath[item],self.labels[0][item]))
        self.samples = sample    
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        length = len(self.samples)
        return length




def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

if __name__ == '__main__':
    liveroot = 'D:\zwx_Project\zwx_IQA\dataset\databaserelease2'
    index = list(range(0,29))
    random.shuffle(index)
    train_index = index[0:round(0.8*29)]
    test_index = index[round(0.8*29):29]
    trainset = LIVEFolder(root = liveroot, loader = default_loader, index = train_index)
    testset = LIVEFolder(root = liveroot, loader = default_loader, index = test_index)
