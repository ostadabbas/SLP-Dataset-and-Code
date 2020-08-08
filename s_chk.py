'''
to check relevant paraemters
'''
from os import path
import numpy as np

# check the  mean and std
fd = 'rstT/mean_std'
mod = 'PM'
pth = path.join(fd, 'mean_{}.npy'.format(mod))
dt = np.load(pth)
print('mean is', dt)

pth = path.join(fd, 'std_{}.npy'.format(mod))
dt = np.load(pth)
print('std is', dt)


