'''
test basics
'''

import numpy as np

arr1 = np.array([[1,2], [3,4]])
arr2 = arr1[None, :, :]
print(arr2)
print(arr2.shape)