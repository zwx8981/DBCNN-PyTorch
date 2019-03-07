import torch.utils.data as data

from PIL import Image

import os
import os.path
#import math
import scipy.io
import numpy as np
import random


def getFileName(path, suffix):
    ''' 获取指定目录下的所有指定后缀的文件名 '''
    filename = []
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
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
        


class LIVEChallengeFolder(data.Dataset):

    def __init__(self, root, loader, index, transform=None, target_transform=None):

        self.root = root
        self.loader = loader
              
        self.imgpath = scipy.io.loadmat(os.path.join(self.root, 'Data', 'AllImages_release.mat'))
        self.imgpath = self.imgpath['AllImages_release']
        self.imgpath = self.imgpath[7:1169]
        self.mos = scipy.io.loadmat(os.path.join(self.root, 'Data', 'AllMOS_release.mat'))
        self.labels = self.mos['AllMOS_release'].astype(np.float32)  
        self.labels = self.labels[0][7:1169]

        sample = []
        
        for i, item in enumerate(index):
            sample.append((os.path.join(self.root, 'Images', self.imgpath[item][0][0]), self.labels[item]))
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
    liveroot = 'D:\zwx_Project\dbcnn_pytorch\dataset\ChallengeDB_release'
    index = list(range(0,1162))
    random.shuffle(index)
    train_index = index[0:round(0.8*1162)]
    test_index = index[round(0.8*1162):1162]
    trainset = LIVEChallengeFolder(root = liveroot, loader = default_loader, index = train_index)
    testset = LIVEChallengeFolder(root = liveroot, loader = default_loader, index = test_index)