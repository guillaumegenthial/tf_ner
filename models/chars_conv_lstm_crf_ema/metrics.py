"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


# Standard
train = [98.55, 98.58, 98.47, 99.40, 98.62]
testa = [94.20, 93.85, 94.28, 94.31, 94.11]
testb = [90.97, 91.25, 91.14, 91.41, 91.02]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))

# EMA
train = [98.45, 98.40, 98.49, 99.44, 98.57]
testa = [94.30, 93.98, 94.32, 94.50, 94.35]
testb = [90.91, 91.21, 91.13, 91.17, 91.22]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
