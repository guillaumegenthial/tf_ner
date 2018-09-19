"""Metrics"""

__author__ = "Guillaume Genthial"

import numpy as np


train = [99.16, 99.07, 98.64, 99.03, 98.38]
testa = [94.53, 94.26, 94.00, 93.86, 93.87]
testb = [91.18, 91.10, 91.33, 91.42, 90.99]

print(np.mean(train), np.std(train))
print(np.mean(testa), np.std(testa))
print(np.mean(testb), np.std(testb))
