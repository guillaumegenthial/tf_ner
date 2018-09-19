"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


# Standard
train = [97.86, 98.93, 98.90, 98.77, 99.04]
testa = [93.14, 93.71, 93.59, 93.91, 93.70]
testb = [90.03, 90.43, 90.18, 90.49, 90.76]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))

# EMA
train = [97.77, 98.98, 98.97, 98.82, 99.00]
testa = [93.40, 93.71, 94.04, 94.06, 93.82]
testb = [90.17, 90.70, 90.45, 90.43, 90.75]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
