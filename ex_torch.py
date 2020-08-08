'''
for torch operation test
'''
import torch
from torchvision.models import vgg
x = torch.arange(8).view(2, 1, 2, 2)       # 60 x 1 x 256  x 256
print(x)
x_flip = x.flip(3)
print(x_flip)       # no problems
# print('if x changed', x)    # not touched
# 4 dim, no prob
