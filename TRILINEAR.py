import torch
import torch.nn as nn
import torch.nn.functional as F

class TRILINEAR(nn.Module):

     def __init__(self, is_vec, input_dim=2048):
         super(TRILINEAR, self).__init__()
         #self.thresh = thresh
         self.is_vec = is_vec
         self.output_dim = input_dim
     def _trilinearpool(self, x):
         batchSize, dim, h, w = x.data.shape
         x = x.reshape(batchSize, dim, h * w)
         #x = 1. / (h * w) * x.bmm(x.transpose(1, 2))
         x_norm = F.softmax(dim=2)
         channel_relation = x_norm.bmm(x.transpose(1, 2)) #inter-channel relationship map
         #channel_relation = F.softmax(channel_relation)
         x = channel_relation.bmm(x) #trilinear attention map: b*c*(h*w)
         return x

     #def _signed_sqrt(self, x):
     #    x = torch.mul(x.sign(), torch.sqrt(x.abs()+self.thresh))
     #    return x

     #def _l2norm(self, x):
     #    x = F.normalize(x)
     #    return x

     def forward(self, x):
         x = self._trilinearpool(x)
         #x = self._signed_sqrt(x)
         if self.is_vec:
             x = x.mean(2).squeeze()
         #x = self._l2norm(x)
         return x
