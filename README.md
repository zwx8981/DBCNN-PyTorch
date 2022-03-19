
# DBCNN-Pytorch
An experimental PyTorch implementation of Blind Image Quality Assessment Using A Deep Bilinear Convolutional Neural Network.

# Purpose
Considering the popularity of PyTorch in academia, we hope this repo can help reseachers in IQA.
This repo will be used as an active codebase for integrating advanced technologies for IQA research.  

# Requirements
PyTorch 0.4+
Python 3.6

# Usage with default setting
python DBCNN.py

If you want to re-train the SCNN, you still need Matlab and original repo https://github.com/zwx8981/BIQA_Project for generating synthetically distorted images.

python SCNN.py

# Citation
@article{zhang2020blind,  
  title={Blind Image Quality Assessment Using A Deep Bilinear Convolutional Neural Network},  
  author={Zhang, Weixia and Ma, Kede and Yan, Jia and Deng, Dexiang and Wang, Zhou},  
  journal={IEEE Transactions on Circuits and Systems for Video Technology},  
  volume={30},  
  number={1},  
  pages={36--47},  
  year={2020}  
}

# Acknowledgement
https://github.com/HaoMood/bilinear-cnn

# A remarkable re-implementation and pre-trained weights are available at https://github.com/chaofengc/IQA-PyTorch. Thanks for their great work !

