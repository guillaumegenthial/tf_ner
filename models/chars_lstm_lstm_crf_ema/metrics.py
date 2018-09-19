"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


# Standard
train = [98.97, 98.21, 98.76, 98.31, 98.64]
testa = [94.39, 93.51, 94.37, 94.22, 94.03]
testb = [91.17, 90.65, 91.19, 90.89, 91.15]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))

# EMA
train = [98.80, 98.16, 98.73, 98.27, 98.58]
testa = [94.37, 93.71, 94.50, 94.08, 94.33]
testb = [91.26, 91.17, 91.14, 91.28, 91.20]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
