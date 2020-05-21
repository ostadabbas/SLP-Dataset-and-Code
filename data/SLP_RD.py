'''
SLP reader, provides basic interface to access single items.
All saved in list or dict of list for reading from upper level.
In collaboration work, common interface should be added in class design.
'''
import utils.utils as ut
import utils.utils_PM as ut_p
import numpy as np
import os
import os.path as path
import scipy.io as sio
import cv2
from skimage import io


def getImg_dsPM(dsFd=r'G:\My Drive\ACLab Shared GDrive\datasetPM\danaLab', idx_subj=1, modality='IR', cov='uncover',
                idx_frm=1):
	'''
	directly get image or array in raw format.
	:param dsFd:
	:param idx_subj:
	:param modality:
	:param cov:
	:param idx_frm:
	:return:
	'''
	npy_nmSet = {'depthRaw', 'PMarray'}  # mainly use depth raw and PM array
	if modality in npy_nmSet:  # npy format
		nmFmt = '{:06}.npy'
		# imgPth = os.path.join(dsFd, '{:05d}'.format(idx_subj), modality, cov, nmFmt.format(idx_frm))
		# img = np.load(imgPth)
		readFunc = np.load
	else:
		nmFmt = 'image_{:06d}.png'
		readFunc = io.imread
	imgPth = os.path.join(dsFd, '{:05d}'.format(idx_subj), modality, cov, nmFmt.format(idx_frm))
	img = readFunc(imgPth)  # should be 2d array
	img = np.array(img)
	return img


class SLP_RD:  # slp reader
	# human body definition
	joint_num = 17  # for std
	joint_num_ori = 14  # truth labeled jts,
	joints_name = (
		"R_Ankle", "R_Knee", "R_Hip", "L_Hip", "L_Knee", "L_Ankle", "R_Wrist", "R_Elbow", "R_Shoulder", "L_Shoulder",
		"L_Elbow", "L_Wrist", "Thorax", "Head", "Pelvis", "Torso",
		"Neck")  # max std joints, first joint_num_ori will be true labeled
	evals_name = joints_name[:joint_num_ori + 1]  # original plus one more, pelvis is included for center alignment
	flip_pairs_name = (
		('R_Hip', 'L_Hip'), ('R_Knee', 'L_Knee'), ('R_Ankle', 'L_Ankle'),
		('R_Shoulder', 'L_Shoulder'), ('R_Elbow', 'L_Elbow'), ('R_Wrist', 'L_Wrist')
	)
	skels_name = (
		# ('Pelvis', 'Thorax'),
		('Thorax', 'Head'),
		('Thorax', 'R_Shoulder'), ('R_Shoulder', 'R_Elbow'), ('R_Elbow', 'R_Wrist'),
		('Thorax', 'L_Shoulder'), ('L_Shoulder', 'L_Elbow'), ('L_Elbow', 'L_Wrist'),
		# ('Pelvis', 'R_Hip'),
		('R_Hip', 'R_Knee'), ('R_Knee', 'R_Ankle'),
		# ('Pelvis', 'L_Hip'),
		('L_Hip', 'L_Knee'), ('L_Knee', 'L_Ankle'),
	)
	skels_idx = ut.nameToIdx(skels_name, joints_name=joints_name)

	def __init__(self, opts, phase='train'):
		'''
		:param opts:
			phase [train|test]
			sz_pch = 255  or 288 (model)
			SLP_fd = the folder of dana or sim
		'''
		dsFd = opts.SLP_fd
		self.dsFd = dsFd
		n_frm = 45  # fixed
		if 'simLab' in dsFd:
			n_subj = 7
			d_bed = 2.264  # as meter
			n_split = 0  # the split point of train and test (sim for test only)
		else:
			n_subj = 102
			d_bed = 2.101
			n_split = 90

		self.n_subj = n_subj
		self.n_split = n_split
		self.d_bed = d_bed
		self.sz_depth = [424, 512]
		self.sz_PM = [84, 192]
		self.sz_IR = [120, 160]
		self.sz_RGB = [576, 1024]
		# camera parameters of kinect, make to vertical one
		self.c_d = [208.1, 259.7]
		self.f_d = [367.8, 367.8]
		self.PM_max = 94  # 94 kpa, from 90 percentil
		self.phase = phase
		self.sz_pch = opts.sz_pch
		self.fc_depth = opts.fc_depth
		# mods_src = opt.mods_src
		# mods_tar = opt.mods_tar
		# self.mods_src = mods_src
		# self.mods_tar = mods_tar

		if 'train' == phase:        # default train  / or test
			idxs_subj = range(n_split)
		else:
			idxs_subj = range(n_split, n_subj)
		self.idxs_subj = idxs_subj
		idxs_subj_all = range(n_subj)

		# self.dct_li_PTr = ut_p.genPTr_dict(idxs_subj_all, opt.mod_src + opt.mod_tar, dsFd)
		# get all mdoe
		self.dct_li_PTr = ut_p.genPTr_dict(idxs_subj_all, ['RGB', 'IR', 'depth'], dsFd)
		phys_arr = np.load(path.join(dsFd, 'physiqueData.npy'))
		phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
		# wt(kg)
		# height,
		# gender(0 femal, 1 male),
		# bust (cm),waist ,
		# hip,
		# right upper arm ,
		# right lower arm,
		# righ upper leg,
		# right lower leg
		self.phys_arr = phys_arr.astype(np.float)  # all list

		# for caliPM_li    all list
		caliPM_li = []
		for i in idxs_subj_all:
			pth_cali = os.path.join(dsFd, '{:05d}'.format(i + 1), 'PMcali.npy')
			caliPM = np.load(pth_cali)
			caliPM_li.append(caliPM)
		self.caliPM_li = caliPM_li  # all cali files in order
		# gen the descriptor list   [[ i_subj,   cov,  i_frm  ]]
		pthDesc_li = []  # pth descriptor, make abs from 1,  ds_phase list

		self.li_joints_gt_RGB = []  # for list generation
		self.li_joints_gt_IR = []
		## PM, depth homo estimated
		self.li_joints_gt_depth = []  # fake one
		self.li_joints_gt_PM = []

		# make bb
		self.li_bb_RGB = []
		self.li_bb_IR = []
		self.li_bb_depth = []
		self.li_bb_PM = []

		# joints_gt = sio.loadmat(os.path.join(dsFd, '{:05d}'.format(idx_sub), joints_gt))['joints_gt'][:, :, idx_frm - 1]
		# the PTr trans
		# PTr_src2tar = np.dot(np.linalg.inv(PTr_tar), PTr_src)
		# PTr_src2tar = PTr_src2tar / np.linalg.norm(PTr_src2tar)
		for i in idxs_subj_all:
			# joints_gt_RGB_t = sio.loadmat(os.path.join(dsFd, '{:05d}'.format(i+1), 'joints_gt_RGB.mat'))['joints_gt'].transpose([2,1,0])    # 3 x n_jt x n_frm -> n_jt x 3
			joints_gt_RGB_t = sio.loadmat(os.path.join(dsFd, '{:05d}'.format(i + 1), 'joints_gt_RGB.mat'))[
				'joints_gt']  # 3 x n_jt x n_frm -> n_jt x 3
			# print('joints gt shape', joints_gt_RGB_t.shape)		# check-----------
			joints_gt_RGB_t = joints_gt_RGB_t.transpose([2, 1, 0])
			joints_gt_IR_t = sio.loadmat(os.path.join(dsFd, '{:05d}'.format(i + 1), 'joints_gt_IR.mat'))[
				'joints_gt'].transpose([2, 1, 0])
			# homography RGB to depth
			PTr_RGB = self.dct_li_PTr['RGB'][i]  # default to PM
			PTr_depth = self.dct_li_PTr['depth'][i]
			PTr_RGB2depth = np.dot(np.linalg.inv(PTr_depth), PTr_RGB)
			PTr_RGB2depth = PTr_RGB2depth / np.linalg.norm(PTr_RGB2depth)
			# joints_gt_depth_t = cv2.perspectiveTransform(joints_gt_RGB_t[:, :, :2], PTr_RGB2depth)[0]       # why this operation?
			joints_gt_depth_t = np.array(list(
				map(lambda x: cv2.perspectiveTransform(np.array([x]), PTr_RGB2depth)[0], joints_gt_RGB_t[:, :, :2])))
			joints_gt_depth_t = np.concatenate([joints_gt_depth_t, joints_gt_RGB_t[:, :, 2, None]], axis=2)
			# print('after concatenate',  joints_gt_depth_t.shape)	# check -----
			joints_gt_PM_t = np.array(list(
				map(lambda x: cv2.perspectiveTransform(np.array([x]), PTr_RGB)[0], joints_gt_RGB_t[:, :, :2])))
			joints_gt_PM_t = np.concatenate([joints_gt_PM_t, joints_gt_RGB_t[:, :, 2, None]], axis=2)

			# update li
			self.li_joints_gt_RGB.append(joints_gt_RGB_t)  # n_subj x [ n_frm x n_jt x 3]
			self.li_joints_gt_IR.append(joints_gt_IR_t)
			self.li_joints_gt_depth.append(joints_gt_depth_t)
			self.li_joints_gt_PM.append(joints_gt_PM_t)
			self.li_bb_RGB.append(np.array(list(map(ut.get_bbox, joints_gt_RGB_t))))  # n_subj x 45 x4
			self.li_bb_IR.append(np.array(list(map(ut.get_bbox, joints_gt_IR_t))))
			self.li_bb_depth.append(np.array(list(map(ut.get_bbox, joints_gt_depth_t))))
			self.li_bb_PM.append(np.array(list(map(ut.get_bbox, joints_gt_PM_t))))  # only upper level is list

		for i in idxs_subj:
			for cov in opts.cov_li:  # add pth Descriptor
				for j in range(n_frm):
					pthDesc_li.append([i + 1, cov, j + 1])  # file idx 1 based,  sample idx 0 based
		self.pthDesc_li = pthDesc_li
		self.n_smpl = len(pthDesc_li)

	def get_array_joints(self, idx_smpl=0, mod='depthRaw'):
		'''
		index sample function in with flattened order with given modalities. It could be raw form array or image, so we call it array.
		corresponding joints will be returned too.  depth and PM are perspective transformed. bb also returned
		:param idx_smpl: the index number base 0
		:param mod:   4 modality including raw data
		:return:
		'''
		id_subj, cov, id_frm = self.pthDesc_li[idx_smpl]    # id for file , base 1
		arr = getImg_dsPM(dsFd=self.dsFd, idx_subj=id_subj, modality=mod, cov=cov, idx_frm=id_frm)

		# get unified jt name, get rid of raw extenstion
		if 'depth' in mod:
			mod = 'depth'
		if 'IR' in mod:
			mod = 'IR'
		joints_gt = getattr(self, 'li_joints_gt_{}'.format(mod))
		jt = joints_gt[id_subj-1][id_frm-1]
		bb = getattr(self, 'li_bb_{}'.format(mod))[id_subj-1][id_frm-1]
		return arr, jt, bb

	def bb2ori(self, jts, mod='depth'):
		'''
		recover the joints corresponding to idxs to original image space.
		3rd will
		:param jts:
		:param idxs:
		:return:
		'''

		if 'depth' in mod:
			mod = 'depth'
		if 'IR' in mod:
			mod = 'IR'
		bbs_in = getattr(self, 'li_bb_{}'.format(mod))

		## use exact index
		# bbs = []  # flattened corresponding bbb
		# for idx in idxs:
		# 	n_subj, cov, n_frm = self.pthDesc_li[idx]
		# 	bbs.append(bbs_in[n_subj-1][n_frm-1])

		## all bbs
		bbs = []  # flattened corresponding bbb
		for n_subj, cov, n_frm in self.pthDesc_li:
			bbs.append(bbs_in[n_subj-1][n_frm-1])

		jts_ori = ut.jt_bb2ori(jts, self.sz_pch, bbs)
		if jts_ori.shape[-1]==3:    # if depth jts
			jts_ori[..., -1] = jts_ori[..., -1] /self.fc_depth
		return jts_ori

	def bb2cam(self, jts, mod='depth'):
		'''
		from jts_bb get jts_cam directly.  Actually a wrapper of above functions.
		return both jts cam and jts_ori
		:param jts:
		:param idx:
		:param mod:
		:return:
		'''
		jts_ori = self.bb2ori(jts, mod=mod)
		if 'depth' in mod:
			f= self.f_d
			c = self.c_d
		else:
			print('no recovery for this mod{}'.format(mod))
			exit(-1)
		jts_cam = ut.pixel2cam(jts_ori, f, c)

		return jts_cam, jts_ori