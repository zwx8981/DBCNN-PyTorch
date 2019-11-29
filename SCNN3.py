import os
# import sys
#import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from ImageDataset import ImageDataset
from PIL import Image

from tqdm import tqdm

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class SCNN(nn.Module):

    def __init__(self):
        """Declare all needed layers."""
        super(SCNN, self).__init__()

        # Linear classifier.
        self.inplanes = 128
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.num_class = 125
        backbone = torchvision.models.resnet34(pretrained=True)
        self.shared_features = nn.Sequential(*list(backbone.children())[0:6])
        #self.realistic_head = nn.Sequential(*list(backbone.children())[6:8])
        # self.synthetic_head = nn.Sequential(nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        #                                     nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        #                                     nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        #                                     nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.synthetic_head1 = self._make_layer(BasicBlock, 128, 1, stride=2, dilate=False)
        self.synthetic_head2 = self._make_layer(BasicBlock, 256, 1, stride=2, dilate=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, self.num_class)

        for m in self.synthetic_head1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.synthetic_head2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        weight_init(self.classifier)

        for param in self.shared_features.parameters():
            param.requires_grad = False


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, X):
        #        return X
        X = self.shared_features(X)
        X = self.synthetic_head1(X)
        X = self.synthetic_head2(X)
        X = self.avgpool(X)
        X = self.classifier(X.squeeze())
        return X


class SCNNManager(object):
    """Manager class to train S-CNN.
    """

    def __init__(self, options, path):
        """Prepare the network, criterion, solver, and data.
        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        self._path = path
        self._epoch = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Network.
        network = SCNN()
        #weight_init(network)
        network = network.to(self.device)
        # self._net = network.cuda()
        self._net = network
        #self._net = torch.nn.DataParallel(network)

        logspaced_LR = np.logspace(-1, -4, self._options['epochs'])
        # Load the model from disk.
        checkpoints_list = os.listdir(self._path['model'])
        if len(checkpoints_list) != 0:
            self._net.load_state_dict(torch.load(
                os.path.join(self._path['model'], '%s%s%s' % ('net_params', str(len(checkpoints_list) - 1), '.pkl'))))
            self._epoch = len(checkpoints_list)
            self._options['base_lr'] = logspaced_LR[len(checkpoints_list)]
        # self._net.load_state_dict(torch.load(self._path['model']))
        print(self._net)
        # Criterion.
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Solver.
        self._solver = torch.optim.SGD(
            self._net.parameters(), lr=self._options['base_lr'],
            momentum=0.9, weight_decay=self._options['weight_decay'])
        #        self._solver = torch.optim.Adam(
        #            self._net.parameters(), lr=self._options['base_lr'],
        #            weight_decay=self._options['weight_decay'])
        lambda1 = lambda epoch: logspaced_LR[epoch]
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._solver, lr_lambda=lambda1)

        self.train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=256),  # Let smaller edge match
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        self.test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=256),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        self.train_data = ImageDataset(csv_file=os.path.join(path['kadis'], 'train.txt'),
                                       img_dir=os.path.join(path['kadis'], 'dist_imgs'),
                                       transform=self.train_transforms,
                                       test=False)
        self._train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=1, pin_memory=True)
        self.test_data = ImageDataset(csv_file=os.path.join(path['kadis'], 'test.txt'),
                                       img_dir=os.path.join(path['kadis'], 'dist_imgs'),
                                       transform=self.test_transforms,
                                       test=True)
        self._test_loader = torch.utils.data.DataLoader(
            self.test_data, batch_size=self._options['batch_size'],
            shuffle=False, num_workers=1, pin_memory=True)

    def train(self):
        """Train the network."""
        print('Training.')
        best_acc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tTest acc')
        for t in range(self._epoch, self._options['epochs']):
            epoch_loss = []
            num_correct = 0.0
            num_total = 0.0
            batchindex = 0
            for X, y in tqdm(self._train_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                # y = torch.tensor(y.to(device))

                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
                loss = self._criterion(score, y.detach())
                epoch_loss.append(loss.item())

                # Prediction.
                _, prediction = torch.max(F.softmax(score.data), 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y)
                # Backward pass.
                loss.backward()
                self._solver.step()
                batchindex = batchindex + 1
            print('%d epoch done' % (t + 1))
            train_acc = 100 * num_correct.float() / num_total
            if (t < 2) | (t > 20):
                with torch.no_grad():
                    test_acc = self._accuracy(self._test_loader)
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = t + 1
            print('*', end='')
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
            pwd = os.getcwd()
            modelpath = os.path.join(pwd, 'models', ('net_params' + str(t) + '.pkl'))
            torch.save(self._net.state_dict(), modelpath)
            self._scheduler.step(t)
        print('Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))

    def _accuracy(self, data_loader):
        """Compute the train/test accuracy.
        Args:
            data_loader: Train/Test DataLoader.
        Returns:
            Train/Test accuracy in percentage.
        """
        self._net.eval()
        num_correct = 0.0
        num_total = 0.0
        batchindex = 0
        for X, y in tqdm(data_loader):
            # Data.
            batchindex = batchindex + 1
            X = X.to(self.device)
            y = y.to(self.device)
            # y = torch.tensor(y.to(device))

            # Prediction.
            score = self._net(X)
            _, prediction = torch.max(score.data, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.data)
        self._net.train()  # Set the model to training phase
        return 100 * num_correct.float() / num_total


def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train DB-CNN for BIQA.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-1,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=64, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=30, help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=5e-4, help='Weight decay.')

    args = parser.parse_args()
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
    }

    path = {
        'kadis': '/media/zwx-sjtu/data/kadis700k',
        'model': '/home/zwx-sjtu/codebase/DBCNN-PyTorch-master/models'
    }

    manager = SCNNManager(options, path)
    # manager.getStat()
    manager.train()


if __name__ == '__main__':
    main()
    # network = SCNN().cuda()
    # input = torch.randn(32,3,224,224)
    # input = input.cuda()
    # output = network(input)
    # print(output)