# DBCNN-Pytorch
An experimental PyTorch implementation of Deep Bilinear Pooling for Blind Image Quality Assessment.

# Purpose
Considering the popularity of PyTorch in academia, we hope this repo can help reseachers in IQA.

# Requirements
PyTorch 0.4+
Python 3.6

# Usage with default setting
python DBCNN.py

Only support experiment on LIVE IQA and LIVE Challenge right now, other datasets will be added soon! (I am a busy yet lazy guy...)

If you want to re-train the SCNN, you still need Matlab and original repo https://github.com/zwx8981/BIQA_Project for generating synthetically distorted images.

python SCNN.py

# TODO:
Retraining SCNN without batchnorm layers.

# Acknowledgement
https://github.com/HaoMood/bilinear-cnn


