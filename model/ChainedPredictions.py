import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import model.modules.ChainedPredictions as M
from . import resnet_nc

class ChainedPredictions(nn.Module):
	"""docstring for ChainedPredictions"""
	def __init__(self, modelName, hhKernel, ohKernel, in_ch=3, nJoints=14):
		super(ChainedPredictions, self).__init__()
		self.nJoints = nJoints
		self.modelName = modelName
		self.resnet = getattr(resnet_nc, self.modelName)(pretrained=False, in_ch=in_ch)     # fales one
		self.resnet.avgpool = M.Identity()
		self.resnet.fc = M.Identity()
		self.hiddenChans = 64 ### Add cases!

		self.hhKernel = hhKernel
		self.ohKernel = ohKernel

		self.init_hidden = nn.Conv2d(512, self.hiddenChans, 1)
		_deception = []
		for i in range(self.nJoints):
			_deception.append(M.Deception(self.hiddenChans))
		self.deception = nn.ModuleList(_deception)

		_h2h = []
		_o2h = []
		for i in range(nJoints):
			_o = []
			_h2h.append(
				nn.Sequential(
					nn.Conv2d(self.hiddenChans, self.hiddenChans, kernel_size=self.hhKernel, padding=self.hhKernel//2),
					nn.BatchNorm2d(self.hiddenChans)
				)
			)
			for j in range(i+1):
				_o.append(nn.Sequential(
						nn.Conv2d(1, self.hiddenChans, 1),
						nn.Conv2d(self.hiddenChans, self.hiddenChans, kernel_size=self.ohKernel, stride=2, padding=self.ohKernel//2),
						nn.BatchNorm2d(self.hiddenChans),
						nn.Conv2d(self.hiddenChans, self.hiddenChans, kernel_size=self.ohKernel, stride=2, padding=self.ohKernel//2),
						nn.BatchNorm2d(self.hiddenChans),
					)
				)
			_o2h.append(nn.ModuleList(_o))

		self.h2h = nn.ModuleList(_h2h)
		self.o2h = nn.ModuleList(_o2h)

	def forward(self, x):
		# print('input x shape', x.size())
		hidden = [0]*self.nJoints
		output = [None]*self.nJoints
		rst_resnet = self.resnet(x)
		# print('resnet output size', rst_resnet.size())      # 1x131072 fc ?
		hidden[0] += self.resnet(x).reshape(-1, 512, 8, 8)      # output 512 , 8 , 8 ??   why this, image size 8 x8 problem??
		# print('hidden 0 size',  hidden[0].size())
		hidden[0] = self.init_hidden(hidden[0]) # batch size changed, 240 64 8 8
		# print('hidden 0 size after init', hidden[0].size())
		output[0] = self.deception[0](hidden[0])

		for i in range(self.nJoints-1):
			hidden[i+1] = self.h2h[i](hidden[i])
			for j in range(i+1):
				hidden[i+1] += self.o2h[i][j](output[j])
			hidden[i+1] = torch.relu(hidden[i+1])
			output[i+1] = self.deception[i+1](hidden[i+1])
		rst = torch.cat(output, 1)
		# print('rst shape', rst.size())
		# return torch.cat(output, 1)
		return rst


def get_pose_net(in_ch, out_ch, **kwargs):
	# num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
	# block_class, layers = resnet_spec[num_layers]
	# model = ChainedPredictions('resnet34', self.opts.hhKernel, self.opts.ohKernel, self.opts.nJoints)
	model = ChainedPredictions('resnet34', 1, 1, in_ch=in_ch, nJoints=out_ch)    # default init
	# if is_train and cfg.MODEL.INIT_WEIGHTS:
	# model.init_weights(cfg.MODEL.PRETRAINED)  # always init weight

	return model