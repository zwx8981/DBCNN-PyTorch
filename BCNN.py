'''
@file: BCNN.py
@author: Jiangtao Xie
@author: Peihua Li
'''
import torch
import torch.nn as nn

class BCNN(nn.Module):
     """Bilinear Pool
        implementation of Bilinear CNN (BCNN)
        https://arxiv.org/abs/1504.07889v5
     Args:
         thresh: small positive number for computation stability
         is_vec: whether the output is a vector or not
         input_dim: the #channel of input feature
     """
     def __init__(self, thresh=1e-8, is_vec=True, input_dim=2048):
         """
         Initialize the network.

         Args:
             self: (todo): write your description
             thresh: (float): write your description
             is_vec: (bool): write your description
             input_dim: (int): write your description
         """
         super(BCNN, self).__init__()
         self.thresh = thresh
         self.is_vec = is_vec
         self.output_dim = input_dim * input_dim
     def _bilinearpool(self, x):
         """
         Bmminear pooling.

         Args:
             self: (todo): write your description
             x: (array): write your description
         """
         batchSize, dim, h, w = x.data.shape
         x = x.reshape(batchSize, dim, h * w)
         x = 1. / (h * w) * x.bmm(x.transpose(1, 2))
         return x

     def _signed_sqrt(self, x):
         """
         Return the signed error.

         Args:
             self: (todo): write your description
             x: (array): write your description
         """
         x = torch.mul(x.sign(), torch.sqrt(x.abs()+self.thresh))
         return x

     def _l2norm(self, x):
         """
         Normalize x.

         Args:
             self: (todo): write your description
             x: (array): write your description
         """
         x = nn.functional.normalize(x)
         return x

     def forward(self, x):
         """
         Forward computation of forward computation.

         Args:
             self: (todo): write your description
             x: (todo): write your description
         """
         x = self._bilinearpool(x)
         x = self._signed_sqrt(x)
         if self.is_vec:
             x = x.view(x.size(0),-1)
         x = self._l2norm(x)
         return x