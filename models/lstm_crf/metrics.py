"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [98.78, 98.45, 99.05, 98.97, 99.00]
testa = [93.61, 93.81, 93.73, 93.76, 93.49]
testb = [90.34, 90.61, 90.39, 90.33, 90.43]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
