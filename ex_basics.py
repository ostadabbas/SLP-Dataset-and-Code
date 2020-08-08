'''
test basics
'''

import numpy as np
import re

# arr1 = np.array([[1,2], [3,4]])
# arr2 = arr1[None, :, :]
# print(arr2)
# print(arr2.shape)
#
# ## numpy test
# arr1 = np.array([0,2,5])
# if arr1 is not None:
# 	print('not none')
# else:
# 	print('is none')

## re test
# li_intInStr = [int(s) for s in re.findall(r'\b\d+\b', 'he33llo 42 I\'m a 32 string 30')]  # working fine
# li_intInStr = [int(s) for s in re.findall(r'\d+', 'image_00091_00001')]
# print(li_intInStr)

## parameter calculation
# get the norm of the vector
# eye_dft = [1.0, -1.25, 0.7]
# v_norm_cam = np.linalg.norm(eye_dft)
# print('cam dist is', v_norm_cam)

## basics
# li1= ['haha', 'mama']
# li2 = ['a', *li1,  'b']
# # print(li2)
# print(li1[3:])      # empty
# li1 = 1:2:9
# print(li1)
# a = [1,1 ]
# if a == [1, 1]:
#     print('a equals to [1, 1]')
# else:
#     print('failed')

## numpy
# a = np.array([[1,2,5],
#               [6, 8, 10]])
# a = [[1,2], [3, 4]]
# print(a[:][1])  # eplipsis
# print(*a)   #star operation on numpy , worked
# b = np.arange(18).reshape(3,3,2)
# print(b)
# print(a[..., -1])
# print(a[..., None, -1].shape)       # 2,1
# print(a[..., -1, None].shape)       # 2,1
# print(np.zeros(2))
# print(a[...,None,None,  -1])
# print(a[-1,-1])     # all last index.
# b = np.array([1, 0])
# c = np.array([2,4, 6])
# print(a*c)      # the last dim similar can be broadcast
# print(a*b)      # can't broadcast
# print(a*b[...,None])    # worked
# print(a*np.tile(b.T, [1,2]))  # not working, can't transpose vector
# print(a[: ,0])   # reduce to single dim
# print(b[...,1]) # is ... good for additional dimensions? yes.

## format
# how to format in format string
# str1 = "{:>8}" + "\{:>{:d}}".format(5) * 2        # out of range error
# str1 = "{:>8}" + "\{:>{:d}\}".format(5) * 2
# str1 = "{:>8}" + ("{:>" + "{:d}".format(3) +"}")* 2
# print(str1)
