'''
Feeder of the SLP dataset.
'''
from torch.utils.data.dataset import Dataset
import utils.utils as ut
import torch
import torchvision.transforms as transforms
import numpy as np


class SLP_FD(Dataset):
	# function dict for gettin function
	def __init__(self, ds, opts, phase='train', df_cut=0.03):
		'''
		:param ds:
		:param opts:
		:param phase:
		:param df_cut: the cut thresh above bed
		'''
		self.phase = phase  # train data can be also used for test to generate result
		self.ds = ds  # all data in
		self.sz_pch = opts.sz_pch
		self.prep = opts.prep  # preprocessing method
		self.fc_depth = ds.fc_depth
		self.opts = opts  # for uniform func design in case more parameters are needed
		# define the getData func according to prep  methods for different  jobs
		dct_func_getData = {
			'SLP_A2J': self.SLP_A2J
		}
		self.func_getData = dct_func_getData[opts.prep]

		# special part for A2J
		ct_A2J = 2.9    # the pelvis center
		d_bed = self.ds.d_bed
		d_cut = d_bed - df_cut  # around human body center
		self.bias_A2S = d_cut - ct_A2J     #

	# can define transforms here based on train (aug) or test
	def SLP_A2J(self, idx, if_rt=False):  # outside to costly
		'''
		Pre func. All major preprocessing will be here. __getItem__ only a wrapper so no operations there.
		from SLP ds get all the model(itop) compabible format for training. There seems a counter clockwise operatoion in A2J

		will transform into the C x H xW format
		:param ds:
		:return:
		'''
		# param
		df_cut = 0.04  # 1 cm cut   turn later
		std = 0.54
		# define the fill and cut part
		d_bed = self.ds.d_bed
		d_cut = d_bed - df_cut      # around human body center
		mean = d_cut + 0.5
		bg = d_cut + 0.75  #
		# get depth, joints, bb
		depthR, joints_gt, bb = self.ds.get_array_joints(idx, mod='depthRaw')  # raw depth
		depthM = depthR/1000.
		depthM[depthM>d_cut] = bg       # cut everything to bg
		depth = (depthM - mean) / std   # norm then factor it
		sz_pch = self.opts.sz_pch
		depth, jt = ut.get_patch(depth, joints_gt, bb, sz_pch)  # not keep ratio
		jt[:, 2] = jt[:, 2] * self.fc_depth # to pixel  *50
		if if_rt:       # counter
			depth = np.rot90(depth)      # rotate to another direction
			jt_T = jt.copy()
			jt_T[:, 0] = jt[:,1]
			jt_T[:, 1] = sz_pch[0] - jt[:,0]
			jt = jt_T
		# single raw or
		if depth.ndim < 3:  # gray or depth
			depth = depth[None, :, :]   # add channel to C H W
		else:
			depth = depth.transpose([2, 0, 1]) # RGB to  CHW
		return depth, jt

	def A2J_postProc(self, rst, if_rt = False, if_bia_d = True, if_tr=True):
		'''
		post process the joints from the network. Rotate if needed, add bias to make depth correct.
		put depth to meter
		:param preds:  N x n_jt x 3  ( 12 x 15 x 3)   np
		:param if_rt: if rotate the joints (clockwise)
		:param if_tr: if transpose the coordinate. Seems x, y flip with each other.
		:return:
		'''
		rst_T = rst.copy()
		if if_rt:
			rst_T[:,:,0] = self.opts.sz_pch[1] - rst[:,:,1]
			rst_T[:,:, 1] = rst[:, :, 0]
		if if_tr:
			rst_T[:,:,0] = rst[:,:,1]
			rst_T[:, :, 1] = rst[:, :, 0]

		rst_T[:, :, 2] = rst_T[:, :, 2] / self.fc_depth # recover
		if if_bia_d:    # if there is bias for the depth correction
			rst_T[:, :, 2] = rst[:, :, 2] + self.bias_A2S

		return rst_T

	def __getitem__(self, index):
		arr, jt = self.func_getData(index)

		# transform is for  Image  to Tensor, will change order !! and range 255 - 1.0
		# arr_tch = self.transform(arr)
		# jt_tch = self.transform(jt)
		arr_tch = torch.from_numpy(arr.copy())  # the rotate make negative index along dim
		jt_tch = torch.from_numpy(jt.copy())
		if False:
			print('arr shape', arr.shape)
			print('arr tch shape', arr_tch.size())

		return arr_tch, jt_tch

	def __len__(self):
		return self.ds.n_smpl

