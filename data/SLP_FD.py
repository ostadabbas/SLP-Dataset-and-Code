'''
Feeder of the SLP dataset.
'''
import utils.utils as ut
from data.SLP_RD import uni_mod
from utils.utils_ds import *
from os import path
from utils import vis

class SLP_FD(Dataset):
	# function dict for gettin function
	def __init__(self, ds, opts, phase='train', df_cut=0.03, if_sq_bb = False):
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
			'SLP_A2J': self.SLP_A2J,
			'MPPE3D': self.MPPE3D,
			'jt_hm': self.jt_hm
		}
		self.func_getData = dct_func_getData[opts.prep] # preparastioin
		self.if_sq_bb = if_sq_bb    # if use sq bb

		# special part for A2J
		ct_A2J = 2.9    # the pelvis center
		d_bed = self.ds.d_bed
		d_cut = d_bed - df_cut  # around human body center
		self.bias_A2S = d_cut - ct_A2J     #
		self.d_cut = d_cut
		self.df_cut = df_cut

	# can define transforms here based on train (aug) or test
	def SLP_A2J(self, idx, if_rt=False):  # outside to costly
		'''
		Pre func. All major preprocessing will be here. __getItem__ only a wrapper so no operations there. All ds inteface provides final ts format , return dict rst with data required by the specific models.
		from SLP ds get all the model(itop) compabible format for training. There seems a counter clockwise operatoion in A2J
		will transform into the C x H xW format
		right now , there is only patch and jt,  may add vis or if_depth later
		:param rt_xy: the x,y ratio of the bb
		:return:
		'''
		# param
		std = 0.54
		# define the fill and cut part
		d_cut = self.d_cut
		mean = d_cut + 0.5
		bg = d_cut + 0.75  #
		# get depth, joints, bb
		depthR, joints_gt, bb = self.ds.get_array_joints(idx, mod='depthRaw', if_sq_bb=self.if_sq_bb)  # raw depth
		depthM = depthR/1000.
		depthM[depthM>d_cut] = bg       # cut everything to bg
		depth = (depthM - mean) / std   # norm then factor it
		sz_pch = self.opts.sz_pch

		# the simple version similar to A2J official
		# depth, jt = ut.get_patch(depth, joints_gt, bb, sz_pch)  # not keep ratio
		scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False
		img_patch, trans = ut.generate_patch_image(depth, bb, do_flip, scale, rot, do_occlusion, sz_std=[288, 288] )# use the
		jt = joints_gt.copy()
		for i in range(len(joints_gt)):  # 2d first for boneLen calculation
			jt[i, 0:2] = ut.trans_point2d(joints_gt[i, 0:2], trans)  # to pix:patch under input_shape size

		jt[:, 2] = jt[:, 2] * self.fc_depth # to pixel  *50

		if if_rt:       # counter
			img_patch = np.rot90(img_patch)      # rotate to another direction
			jt_T = jt.copy()
			jt_T[:, 0] = jt[:,1]
			jt_T[:, 1] = sz_pch[0] - jt[:,0]
			jt = jt_T
		# single raw or
		if img_patch.ndim < 3:  # gray or depth
			img_patch = img_patch[None, :, :]   # add channel to C H W
		else:
			img_patch = img_patch.transpose([2, 0, 1]) # RGB to  CHW
		# to tensor
		arr_tch = torch.from_numpy(img_patch.copy())  # the rotate make negative index along dim
		jt_tch = torch.from_numpy(jt.copy())
		bb_tch = torch.from_numpy(bb)
		rst = {'arr_tch':arr_tch, 'jt_tch':jt_tch, 'bb_tch':bb_tch}
		return rst

	def A2J_postProc(self, rst, np_bb, if_rt = False, if_bia_d = True, if_tr=True):
		'''
		post process the joints from the network. Rotate if needed, add bias to make depth correct.
		put depth to meter [x,y:pix(ori), z(meter)]. only change d to proper meter
		:param preds:  N x n_jt x 3  ( 12 x 15 x 3)   np
		:param if_rt: if rotate the joints (clockwise)
		:param if_tr: if transpose the coordinate. Seems x, y flip with each other.
		:return:
		:history:  6/11/20,  make it to ori space,  with d in metter
		'''
		rst_T = rst.copy()
		if if_rt:   # sus rotation
			rst_T[:,:,0] = self.opts.sz_pch[1] - rst[:,:,1]
			rst_T[:,:, 1] = rst[:, :, 0]
		if if_tr:       # suspecious transpose
			rst_T[:,:,0] = rst[:,:,1]
			rst_T[:, :, 1] = rst[:, :, 0]
		preds_ori = rst_T.copy()
		preds_ori[:, :, 2] = preds_ori[:, :, 2] / self.fc_depth # recover
		if if_bia_d:    # if there is bias for the depth correction
			preds_ori[:, :, 2] = preds_ori[:, :, 2] + self.bias_A2S
		preds_ori = ut.jt_bb2ori(preds_ori, self.sz_pch, np_bb)
		return rst_T, preds_ori

	def MPPE3D(self, idx):
		# use  3DMPPE given mean and stad
		pixel_mean = np.array((0.485, 0.456, 0.406))
		pixel_std = np.array((0.229, 0.224, 0.225))

		arr, joints_gt, bb = self.ds.get_array_joints(idx, mod='RGB', if_sq_bb=True)  # raw depth
		if not self.opts.if_bb:
			sz_pch = tuple(self.opts.sz_pch)
			sz_ori = self.ds.sz_RGB
			bb=[0, 0, sz_ori[0], sz_ori[1]]     # full image bb , make square bb
			bb = ut.adj_bb(bb, rt_xy=1)

		# get patch
		scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False
		img_patch, trans = ut.generate_patch_image(arr, bb, do_flip, scale, rot, do_occlusion)      # default std
		if arr.ndim >2:
			img_channels = 3
			for i in range(img_channels):
				img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)  # 255 range

		trm = transforms.Compose([transforms.ToTensor(),
			transforms.Normalize(mean=pixel_mean, std=pixel_std)]
			)
		arr_tch = trm(img_patch)  # get the CHW format 0~1
		bb_tch = torch.from_numpy(bb)       # get the bb directly
		rst = {'arr_tch':arr_tch, 'bb_tch':bb_tch}
		# jt should be normalized to hm size 64, for test we don't change that
		return rst
	def jt_hm(self, idx):
		'''
		joint heatmap format feeder.  get the img, hm(gaussian),  jts, l_std (head_size)
		:param index:
		:return:
		'''
		mods = self.opts.mod_src
		n_jt = self.ds.joint_num_ori    # use ori joint
		sz_pch = self.opts.sz_pch
		out_shp = self.opts.out_shp[:2]
		ds = self.ds
		mod0 = mods[0]
		li_img = []
		li_mean =[]
		li_std = []
		img, joints_ori, bb = self.ds.get_array_joints(idx, mod=mod0, if_sq_bb=True)  # raw depth    # how transform
		joints_ori = joints_ori[:n_jt, :2]  # take only the original jts

		img_height, img_width = img.shape[:2]       #first 2
		if not self.opts.if_bb:
			# sz_ori = self.ds.sz_RGB
			mod_unm = uni_mod(mod0)
			sz_ori = ds.sizes[mod_unm]
			bb = [0, 0, sz_ori[0], sz_ori[1]]  # full image bb , make square bb
			bb = ut.adj_bb(bb, rt_xy=1) # get sqrt from ori size

		li_mean += self.ds.means[mod0]
		li_std += self.ds.stds[mod0]

		if not 'RGB' in mods[0]:
			img = img[..., None]        # add one dim
		li_img.append(img)

		for mod in mods[1:]:
			img = self.ds.get_array_A2B(idx=idx, modA=mod, modB=mod0)
			li_mean += self.ds.means[mod]
			li_std += self.ds.stds[mod]
			if 'RGB' != mod:
				img = img[...,None]     # add dim
			li_img.append(img)
		img_cb = np.concatenate(li_img, axis=-1)    # last dim, joint mods

		# augmetation
		# 2. get augmentation params
		if self.phase=='train':     # only aug when train needed
			scale, rot, do_flip, color_scale, do_occlusion = get_aug_config()
		else:
			scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False
		# do_flip = False  #         force flip to check
		# do_occlusion = False
		# rot = 60.
		# 3. crop patch from img and perform data augmentation (flip, rot, color scale, synthetic occlusion)
		# print('img_cb shape is', img_cb.shape)
		img_patch, trans = generate_patch_image(img_cb, bb, do_flip, scale, rot, do_occlusion, input_shape=self.opts.sz_pch[::-1])   # ori to bb, flip first trans later
		if img_patch.ndim<3:
			img_channels = 1        # add one channel
			img_patch = img_patch[..., None]
		else:
			img_channels = img_patch.shape[2]   # the channels
		for i in range(img_channels):
			img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)

		# 4. generate patch joint ground truth
		# flip joints and apply Affine Transform on joints
		joints_pch = joints_ori.copy()      # j

		if do_flip:
			joints_pch[:, 0] = img_width - joints_ori[:, 0] - 1
			for pair in self.ds.flip_pairs:
				joints_pch[pair[0], :], joints_pch[pair[1], :] = joints_pch[pair[1], :].copy(), joints_pch[pair[0], :].copy()       # maybe not copy issue
		for i in range(len(joints_pch)):  #  jt trans
			joints_pch[i, 0:2] = trans_point2d(joints_pch[i, 0:2], trans)
		stride = sz_pch[0]/out_shp[1]  # jt shrink
		joints_hm = joints_pch/stride

		joints_vis = np.ones(n_jt)      # n x 1
		for i in range(len(joints_pch)):        # only check 2d here
			# joints_ori [i, 2] = (joints_ori [i, 2] + 1.0) / 2.  # 0~1 normalize
			joints_vis[i] *= (
					(joints_pch[i, 0] >= 0) & \
					(joints_pch[i, 0] < self.opts.sz_pch[0]) & \
					(joints_pch[i, 1] >= 0) & \
					(joints_pch[i, 1] < self.opts.sz_pch[1])
			)  # nice filtering  all in range visibile

		hms, jt_wt = generate_target(joints_hm, joints_vis, sz_hm=out_shp[::-1])  # n_gt x H XW
		idx_t, idx_h = ut.nameToIdx(('Thorax', 'Head'), ds.joints_name)
		l_std_hm = np.linalg.norm(joints_hm[idx_h] - joints_hm[idx_t])
		l_std_ori = np.linalg.norm(joints_ori[idx_h] - joints_ori[idx_t])

		if_vis = False
		if if_vis:
			print('saving feeder data out to rstT')
			tmpimg = img_patch.copy().astype(np.uint8)  # rgb
			tmpkps = np.ones((n_jt, 3))
			tmpkps[:, :2] = joints_pch[:, :2]
			tmpkps[:, 2] = joints_vis
			tmpimg = vis_keypoints(tmpimg, tmpkps, ds.skels_idx)  # rgb
			cv2.imwrite(path.join('rstT', str(idx) + '_pch.jpg'), tmpimg)
			hmImg = hms.sum(axis=0)  #
			# hm_nmd = hmImg.copy()
			# cv2.normalize(hmImg, hm_nmd, beta=255)        # normalize not working
			hm_nmd = ut.normImg(hmImg)
			# print('hms shape', hms.shape)
			cv2.imwrite(path.join('rstT', str(idx) + '_hm.jpg'), hm_nmd)
			tmpimg = ut.normImg(tmpimg)
			img_cb = vis.hconcat_resize([tmpimg, hm_nmd])
			cv2.imwrite(path.join('rstT', str(idx) + '_cb.jpg'), img_cb)

		trans_tch = transforms.Compose([transforms.ToTensor(),
			transforms.Normalize(mean=li_mean, std=li_std)]
			)
		pch_tch = trans_tch(img_patch)
		hms_tch = torch.from_numpy(hms)
		rst = {
			'pch':pch_tch,
			'hms': hms_tch,
			'joints_vis': jt_wt,
			'joints_pch': joints_pch.astype(np.float32),       # in case direct reg
			'l_std_hm':l_std_hm.astype(np.float32),
			'l_std_ori':l_std_ori.astype(np.float32),
			'joints_ori': joints_ori.astype(np.float32),
			'bb': bb.astype(np.float32)     # for recover
		}
		return rst

	def __getitem__(self, index):
		# call the specific processing
		rst = self.func_getData(index)
		return rst

	def __len__(self):
		return self.ds.n_smpl

