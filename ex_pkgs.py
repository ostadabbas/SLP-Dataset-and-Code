'''
test packages functions
'''

import cv2
import numpy as np
import visdom
import matplotlib.pyplot as plt
import json

# img = cv2.imread('testImgs/RGB_001_000001.png')
# imgIR = cv2.imread('testImgs/IR_001_000001.png')
# rows, cols = img.shape[:2]
# M = np.float32([[1, 0, 100], [0, 1, 50]])
# dst_RGB  = cv2.warpAffine(img, M, (cols, rows))
# print('dst RGB shape', dst_RGB.shape)
#
# img_gray = img[:,:,0]   # test gray
# dst_gray = cv2.warpAffine(img_gray, M, (cols, rows))
# print('dst gray shape', dst_gray.shape)
#
# img_c1 = img_gray[..., None] # unsqueeze    single channel will be removed
# print('img_c1 shape', img_c1.shape)
# dst_c1 = cv2.warpAffine(img_c1, M, (cols, rows))
# print('dst c1 shape', dst_c1.shape)
#
# img_c2 = img[...,:2] #  first 2 channel, work on only 1st 2 channels.
# print('img_c2 shape', img_c2.shape)
# dst_c2 = cv2.warpAffine(img_c2, M, (cols, rows))
# print('dst c2 shape', dst_c2.shape)


# cm = getattr(cv2, 'COLORMAP_JET')
# pseu_IR = cv2.applyColorMap(imgIR, cm)
# cv2.imshow('img', pseu_IR)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# visdom test
# vis = visdom.Visdom()
# # vis = visdom.Visdom(server="http://10.99.111.16")       # specify to it
# vis.text('Hello, world!')
# vis.image(np.ones((3, 100, 100)))

# matplotlib

fig, axes = plt.subplots(2, 4, figsize=(25, 12))  # Create a figure containing a single axes.
axes = axes.flatten()
print(type(axes))  # nd_array

# for ax in axes:     # loop through axes worked
# 	ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the axes.

for i in range(len(axes)):      # no problem!
	ax = axes[i]
	ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.show()