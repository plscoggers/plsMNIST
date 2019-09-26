'''
Slightly unecessary but just wanted to create this file to play a bit
'''


import torch
from torchvision import datasets
import torchvision

mnist_data_train = datasets.MNIST('datasets',train=True,download=True)
mnist_data_test = datasets.MNIST('datasets',train=False,download=True)
