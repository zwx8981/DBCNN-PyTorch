import os
#import sys
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import WPFolder
from PIL import Image

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        


class SCNN(nn.Module):

    def __init__(self):
        """Declare all needed layers."""
        super(SCNN, self).__init__()

        # Linear classifier.

        self.num_class = 39
#        self.features = nn.Sequential(nn.Conv2d(3,48,3,1,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(48,48,3,2,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(48,64,3,1,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(64,64,3,2,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(64,64,3,1,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(64,64,3,2,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(64,128,3,1,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(128,128,3,1,1),nn.ReLU(inplace=True),
#                                      nn.Conv2d(128,128,3,2,1),nn.ReLU(inplace=True))
        self.features = nn.Sequential(nn.Conv2d(3,48,3,1,1),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
                                      nn.Conv2d(48,48,3,2,1),nn.BatchNorm2d(48),nn.ReLU(inplace=True),
                                      nn.Conv2d(48,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                      nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,3,2,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        weight_init(self.features)
        self.pooling = nn.AvgPool2d(14,1)
        self.projection = nn.Sequential(nn.Conv2d(128,256,1,1,0), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                        nn.Conv2d(256,256,1,1,0), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        weight_init(self.projection)    
        self.classifier = nn.Linear(256,self.num_class)
        weight_init(self.classifier)

    def forward(self, X):
#        return X
        N = X.size()[0]
        assert X.size() == (N, 3, 224, 224)
        X = self.features(X)
        assert X.size() == (N, 128, 14, 14)
        X = self.pooling(X)
        assert X.size() == (N, 128, 1, 1)
        X = self.projection(X)
        X = X.view(X.size(0), -1)          
        X = self.classifier(X)
        assert X.size() == (N, self.num_class)
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
        # Network.
        network = SCNN()
        weight_init(network)
        #self._net = network.cuda()
        self._net = torch.nn.DataParallel(network).cuda()
        
        logspaced_LR = np.logspace(-1,-4, self._options['epochs'])   
        # Load the model from disk.
        checkpoints_list = os.listdir(self._path['model'])
        if len(checkpoints_list) != 0:
            self._net.load_state_dict(torch.load(os.path.join(self._path['model'],'%s%s%s' % ('net_params', str(len(checkpoints_list)-1), '.pkl'))))
            self._epoch = len(checkpoints_list)
            self._options['base_lr'] = logspaced_LR[len(checkpoints_list)]
        #self._net.load_state_dict(torch.load(self._path['model']))
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
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._solver,lr_lambda=lambda1)

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=256),  # Let smaller edge match
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=256),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        train_data = WPFolder.WPFolder(
            root=self._path['waterloo_pascal'], loader = default_loader, extensions = IMG_EXTENSIONS,
            transform=train_transforms,train = True, ratio = 0.8)
        test_data = WPFolder.WPFolder(
            root=self._path['waterloo_pascal'], loader = default_loader, extensions = IMG_EXTENSIONS,
            transform=test_transforms, train = False, ratio = 0.8)
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=0, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=self._options['batch_size'],
            shuffle=False, num_workers=0, pin_memory=True)

    def train(self):
        """Train the network."""
        print('Training.')
        best_acc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tTest acc')
        for t in range(self._epoch,self._options['epochs']):
            epoch_loss = []
            num_correct = 0.0
            num_total = 0.0
            batchindex = 0
            for X, y in self._train_loader:
               X = torch.tensor(X.cuda())
               y = torch.tensor(y.cuda(async=True))
               #y = torch.tensor(y.to(device))

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
            print('%d epoch done' % (t+1))
            train_acc = 100 * num_correct.float() / num_total               
            if (t < 2) | (t > 20):
                with torch.no_grad():
                    test_acc = self._accuracy(self._test_loader)   
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = t + 1
            print('*', end='')
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
                  (t+1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc)) 
            pwd = os.getcwd()
            modelpath = os.path.join(pwd,'models',('net_params' + str(t) + '.pkl'))
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
        for X, y in data_loader:
            # Data.
            batchindex = batchindex + 1
            X = torch.tensor(X.cuda())
            y = torch.tensor(y.cuda(async=True))
            #y = torch.tensor(y.to(device))

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
                        default=128, help='Batch size.')
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
        'waterloo_pascal': 'Z:\Waterloo\exploration_database_and_code\image',
        'model': 'D:\zwx_Project\dbcnn_pytorch\models'
    }

    manager = SCNNManager(options, path)
    # manager.getStat()
    manager.train()


if __name__ == '__main__':
    main()