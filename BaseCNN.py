import os

import torch
import torchvision
import torch.nn as nn
from Koniq_10k import Koniq_10k
from PIL import Image
from scipy import stats
import random
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from BCNN import BCNN
from MPNCOV import MPNCOV
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)
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
    
    

class BaseCNN(nn.Module):

    def __init__(self, options):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.basemodel = torchvision.models.resnet18(pretrained=True)
        self.options = options
        if self.options['representation'] == 'BCNN':
            self.representation = BCNN(input_dim=512)
            self.fc = nn.Linear(512 * 512, 1)
        elif self.options['representation'] == 'MPNCOV':
            dr = 64
            self.representation = MPNCOV(iterNum=5, input_dim=512, dimension_reduction=dr)
            self.fc = nn.Linear(int(dr*(dr+1)/2), 1)
        else:
            self.fc = nn.Linear(512, 1)
        
        if self.options['fc'] == True:
            # Freeze all previous layers.
            for param in self.basemodel.parameters():
                param.requires_grad = False
            # Initialize the fc layers.
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

        else:
            for param in self.basemodel.conv1.parameters():
                param.requires_grad = False
            for param in self.basemodel.bn1.parameters():
                param.requires_grad = False
            for param in self.basemodel.layer1.parameters():
                param.requires_grad = False
            #for param in self.basemodel.layer2.parameters():
            #    param.requires_grad = False
            #for param in self.basemodel.layer3.parameters():
            #    param.requires_grad = False
        

    def forward(self, X):
        """Forward pass of the network.
        """
        X = self.basemodel.conv1(X)
        X = self.basemodel.bn1(X)
        X = self.basemodel.relu(X)
        X = self.basemodel.maxpool(X)
        X = self.basemodel.layer1(X)
        X = self.basemodel.layer2(X)
        X = self.basemodel.layer3(X)
        X = self.basemodel.layer4(X)
        if self.options['representation'] != None:
            X = self.representation(X)
            if self.options['representation'] == 'MPNCOV':
                X = X.squeeze(2)
        else:
            X = self.basemodel.avgpool(X)
            X = X.squeeze(2).squeeze(2)
        #X = torch.mean(torch.mean(X4,2),2)
        X = self.fc(X)
        return X


class TrainManager(object):
    def __init__(self, options, path):
        """Prepare the network, criterion, solver, and data.
        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        self._path = path

        # Network.
        #self._net = nn.DataParallel(BaseCNN(self._options), device_ids=[0]).cuda()
        self._net = BaseCNN(self._options)

        # Solver.
        if self._options['fc'] == True:
            self._solver = torch.optim.Adam(
                    self._net.parameters(), lr=self._options['base_lr'],
                    weight_decay=self._options['weight_decay'])
            #self._solver = torch.optim.SGD(self._net.parameters(), lr=self._options['base_lr'],
            #                            momentum=0.9,
            #                            weight_decay=self._options['weight_decay'], nesterov=True)
            self._scheduler = torch.optim.lr_scheduler.StepLR(self._solver, step_size=8, gamma=0.1)
        else:
            self._solver = torch.optim.Adam(
                    self._net.parameters(), lr=self._options['base_lr'],
                    weight_decay=self._options['weight_decay'])
            #self._solver = torch.optim.SGD(self._net.parameters(), lr=self._options['base_lr'],
            #                            momentum=0.9,
            #                            weight_decay=self._options['weight_decay'], nesterov=True)
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._solver, milestones=[10, 20, 25], gamma=0.1)
        self._net = torch.nn.DataParallel(self._net, device_ids=[0]).cuda()

        if self._options['fc'] == False:
            self._net.load_state_dict(torch.load(path['fc_root']))

        print(self._net)
        # Criterion.
        if self._options['objective'] == 'l2':
            self._criterion = nn.MSELoss().cuda()
        elif self._options['objective'] == 'l1':
            self._criterion = nn.L1Loss().cuda()
        else:
            self._criterion = nn.SmoothL1Loss().cuda()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize((384, 288)),
            torchvision.transforms.RandomCrop((288, 216)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))])
            
            
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((384, 288)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))])
            
        train_data = Koniq_10k(root=self._path['koniq'], loader=default_loader, index=self._options['train_index'],
                    transform=train_transforms)
        test_data = Koniq_10k(root=self._path['koniq'], loader=default_loader, index=self._options['test_index'],
                    transform=test_transforms)

        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=12, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=self._options['batch_size'],
            shuffle=False, num_workers=12, pin_memory=True)

    def train(self):
        """Train the network."""
        print('Training.')
        best_srcc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self._options['epochs']):
            epoch_loss = []
            pscores = []
            tscores = []
            num_total = 0
            self._scheduler.step()
            for X, y in tqdm(self._train_loader):
                # Data.
                X = X.to(self.device)
                y = y.to(self.device)
                #X = torch.tensor(X.cuda())
                #y = torch.tensor(y.cuda())

                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
                loss = self._criterion(score, y.view(len(score), 1).detach())
                epoch_loss.append(loss.item())
                # Prediction.
                num_total += y.size(0)
                pscores = pscores + score.squeeze(1).cpu().tolist()
                tscores = tscores + y.cpu().tolist()
                # Backward pass.
                loss.backward()
                self._solver.step()
            train_srcc, _ = stats.spearmanr(pscores, tscores)
            with torch.no_grad():
                test_srcc, test_plcc = self._consitency(self._test_loader)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_epoch = t + 1
                print('*', end='')
                pwd = os.getcwd()
                if self._options['fc'] == True:
                    modelpath = os.path.join(pwd, 'fc_models', ('net_params' + '_best' + '.pkl'))
                else:
                    modelpath = os.path.join(pwd, 'db_models', ('net_params' + '_best' + '.pkl'))
                torch.save(self._net.state_dict(), modelpath)

            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t%4.4f' %
                  (t+1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))           

        print('Best at epoch %d, test srcc %f' % (best_epoch, best_srcc))
        return best_srcc

    def _consitency(self, data_loader):
        self._net.train(False)
        num_total = 0
        pscores = []
        tscores = []
        for X, y in tqdm(data_loader):
            # Data.
            X = X.to(self.device)
            y = y.to(self.device)
            #X = torch.tensor(X.cuda())
            #y = torch.tensor(y.cuda())

            # Prediction.
            score = self._net(X)
            #pscores = pscores + score[0].cpu().tolist() #suitable for batchsize=1
            pscores = pscores + score.squeeze(1).cpu().tolist()
            tscores = tscores + y.cpu().tolist()

            #num_total += y.size(0)

        test_srcc, _ = stats.spearmanr(pscores, tscores)
        test_plcc, _ = stats.pearsonr(pscores, tscores)
        self._net.train(True)  # Set the model to training phase
        return test_srcc, test_plcc

def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train CNN for BIQA.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-4,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=24, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=30, help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=5e-4, help='Weight decay.')
    parser.add_argument('--objective', dest='objective', type=str,
                        default='l2', help='l1 | l2 | smoothl1')
    parser.add_argument('--representation', dest='representation', type=str,
                        default=None, help='BCNN | NetVLAD | MPNCOV | None')
    
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
        'objective': args.objective,
        'representation': args.representation,
        'fc': [],
        'train_index': [],
        'test_index': []
    }
    
    path = {
        'koniq': os.path.join('/home/zwx-sjtu/codebase/IQA_database/koniq-10k/'),
        'fc_model': os.path.join('fc_models'),
        'fc_root': os.path.join('fc_models', 'net_params_best.pkl'),
        'db_model': os.path.join('db_models')
    }
    
    index = list(range(0, 10073))
    
    lr_backup = options['base_lr']
    bs_backup = options['batch_size']
    epoch_backup = options['epochs']
    srcc_all = np.zeros((1, 10), dtype=np.float)
    
    for i in range(0, 10):
        #randomly split train-test set
        random.shuffle(index)
        train_index = index[0:round(0.8*len(index))]
        test_index = index[round(0.8*len(index)):len(index)]
    
        options['train_index'] = train_index
        options['test_index'] = test_index
        #train the fully connected layer only
        options['fc'] = True
        options['base_lr'] = 1e-2
        options['batch_size'] = 64
        options['epochs'] = 16
        manager = TrainManager(options, path)
        best_srcc = manager.train()
    
        #fine-tune all model
        options['fc'] = False
        options['base_lr'] = lr_backup
        options['batch_size'] = bs_backup
        options['epochs'] = epoch_backup
        manager = TrainManager(options, path)
        best_srcc = manager.train()
        
        srcc_all[0][i] = best_srcc
        
    srcc_mean = np.mean(srcc_all)
    print(srcc_all)
    print('average srcc:%4.4f' % (srcc_mean))    
    return best_srcc


if __name__ == '__main__':
    main()
