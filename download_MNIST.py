# -*- coding: utf-8 -*-

"""
@author     : zuti
@software   : PyCharm
@file       : download_MNIST.py
@time       : 02/10/2023 22:59
@desc       ：

"""
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，# 并除以255使得所有像素的数值均在0到1之间

trans = transforms.ToTensor()

mnist_train = torchvision.datasets.FashionMNIST(root="../data",
                              train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data",
                              train=False, transform=trans, download=True)