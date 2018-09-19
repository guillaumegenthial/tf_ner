"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [98.98, 99.20, 98.39, 98.81, 98.75]
testa = [94.07, 94.22, 93.78, 94.36, 93.68]
testb = [90.79, 90.86, 91.22, 91.02, 91.14]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
