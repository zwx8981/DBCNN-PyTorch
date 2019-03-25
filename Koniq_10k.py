import torch.utils.data as data

from PIL import Image

import os
import os.path
# import math
import csv
import numpy as np



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


class Koniq_10k(data.Dataset):

    def __init__(self, root, loader, index, transform=None, target_transform=None):

        self.root = root
        self.loader = loader

        self.imgname = []
        self.mos = []
        self.csv_file = os.path.join(self.root, 'koniq10k_scores_and_distributions.csv')
        with open(self.csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.imgname.append(row['image_name'])
                mos = float(row['MOS'])
                mos = np.array(mos)
                mos = mos.astype(np.float32)
                self.mos.append(mos)

        sample = []

        for i, item in enumerate(index):
            sample.append((os.path.join(self.root, '1024x768', self.imgname[item]), self.mos[item]))
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



def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


if __name__ == '__main__':
    print('OK')