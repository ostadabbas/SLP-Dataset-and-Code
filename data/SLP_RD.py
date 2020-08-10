'''
SLP reader, provides basic interface to access single items.
All saved in list or dict of list for reading from upper level.
In collaboration work, common interface should be added in class design.
to add: pre-calculate a fix window for the each ds segment (dana/sim) ?

'''
import utils.utils as ut
import utils.utils_PM as ut_p
import numpy as np
import os
import os.path as path
import scipy.io as sio
import cv2
from skimage import io
from tqdm import tqdm

cov_dict={'uncover': 0, 'cover1': 1, 'cover2': 2}

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
		readFunc = np.load
	else:
		nmFmt = 'image_{:06d}.png'
		readFunc = io.imread
	imgPth = os.path.join(dsFd, '{:05d}'.format(idx_subj), modality, cov, nmFmt.format(idx_frm))
	img = readFunc(imgPth)  # should be 2d array
	img = np.array(img)
	return img


def uni_mod(mod):
	'''
	unify the mod name, so depth and depth raw both share depth related geometry parameters, such as resolution and homography
	:param mod:
	:return:
	'''
	if 'depth' in mod:
		mod = 'depth'
	if 'IR' in mod:
		mod = 'IR'
	if 'PM' in mod:
		mod = 'PM'
	return mod


def genPTr_dict(subj_li, mod_li, dsFd=r'G:\My Drive\ACLab Shared GDrive\datasetPM\danaLab'):
	'''
	loop idx_li, loop mod_li then generate dictionary {mod[0]:PTr_li[...], mod[1]:PTr_li[...]}
	history: 6/3/20: add 'PM' as eye matrix for simplicity
	:param subj_li:
	:param mod_li:
	:return:
	'''
	PTr_dct_li_src = {}  # a dict
	for modNm in mod_li:  # initialize the dict_li
		PTr_dct_li_src[modNm] = []  # make empty list  {md:[], md2:[]...}
	for i in subj_li:
		for mod in mod_li:  # add mod PTr
			mod = uni_mod(mod)  #clean
			if 'PM' not in mod:
				pth_PTr = os.path.join(dsFd, '{:05d}'.format(i + 1), 'align_PTr_{}.npy'.format(mod))
				PTr = np.load(pth_PTr)
			else:
				PTr = np.eye(3)     # fill PM with identical matrix
			PTr_dct_li_src[mod].append(PTr)
	return PTr_dct_li_src

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
	dct_clrMap = {      # the name of cv2 color map
		"depth":'COLORMAP_BONE',
		"depthRaw":'COLORMAP_BONE',
		'IR':'COLORMAP_HOT',
		'IRraw': 'COLORMAP_HOT',
		'PM': 'COLORMAP_JET',
		'PMraw': 'COLORMAP_JET',
	}
	skels_idx = ut.nameToIdx(skels_name, joints_name=joints_name)
	flip_pairs = ut.nameToIdx(flip_pairs_name, joints_name)

	def __init__(self, opts, phase='train', if_0base=True):
		'''
		for simplab,
		:param opts:
			phase [train|test]
			sz_pch = 255  or 288 (model)
			SLP_fd = the folder of dana or sim
		:param if_0base: original is 1 based coord, if set, will change to 0 based format
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
		self.sizes={
			'RGB': [576, 1024],
			'PM': [84, 192],
			'IR': [120, 160],
			'depth': [424, 512]
		}
		self.sz_depth = [424, 512]
		self.sz_PM = [84, 192]
		self.sz_IR = [120, 160]
		self.sz_RGB = [576, 1024]
		# camera parameters of kinect, make to vertical one
		self.c_d = [208.1, 259.7]       # z/f = x_m/x_p so m or mm doesn't matter
		self.f_d = [367.8, 367.8]
		self.PM_max = 94  # 94 kpa, from 90 percentil
		self.phase = phase
		self.sz_pch = opts.sz_pch
		self.fc_depth = opts.fc_depth

		self.means={
			'RGB': [0.3875689, 0.39156103, 0.37614644],
			'depth': [0.7302197],
			'depthRaw': [2190.869],
			'IR': [0.1924838],
			'PM': [0.009072126],
		}
		self.stds = {
			'RGB': [0.21462509, 0.22602762, 0.21271782],
			'depth': [0.25182092],
			'depthRaw': [756.1536],
			'IR': [0.077975444],
			'PM': [0.038837425],
		}

		if 'train' == phase:        # default train  / or test
			idxs_subj = range(n_split)
		elif 'test' == phase:
			idxs_subj = range(n_split, n_subj)
		elif 'all' == phase:
			idxs_subj = range(n_subj)   # all subj

		self.idxs_subj = idxs_subj
		idxs_subj_all = range(n_subj)
		if 'simLab' in dsFd:        # for simLab all splits the same
			idxs_subj = idxs_subj_all
			self.idxs_subj= idxs_subj_all       # for simlab, gives all data as train also as samples can't be 0 in loader.

		# get all mdoe
		self.dct_li_PTr = genPTr_dict(idxs_subj_all, ['RGB', 'IR', 'depth', 'PM'], dsFd)
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
		self.li_bb_sq_RGB = []
		self.li_bb_sq_IR = []
		self.li_bb_sq_depth = []
		self.li_bb_sq_PM = []

		self.li_caliPM = []


		for i in tqdm(idxs_subj_all, desc='initializing SLP'):
			joints_gt_RGB_t = sio.loadmat(os.path.join(dsFd, '{:05d}'.format(i + 1), 'joints_gt_RGB.mat'))[
				'joints_gt']  # 3 x n_jt x n_frm -> n_jt x 3
			# print('joints gt shape', joints_gt_RGB_t.shape)		# check-----------
			joints_gt_RGB_t = joints_gt_RGB_t.transpose([2, 1, 0])
			joints_gt_IR_t = sio.loadmat(os.path.join(dsFd, '{:05d}'.format(i + 1), 'joints_gt_IR.mat'))[
				'joints_gt'].transpose([2, 1, 0])
			if if_0base:
				joints_gt_RGB_t = joints_gt_RGB_t - 1  # to 0 based
				joints_gt_IR_t = joints_gt_IR_t - 1
			# homography RGB to depth
			PTr_RGB = self.dct_li_PTr['RGB'][i]  # default to PM
			PTr_depth = self.dct_li_PTr['depth'][i]
			PTr_RGB2depth = np.dot(np.linalg.inv(PTr_depth), PTr_RGB)
			PTr_RGB2depth = PTr_RGB2depth / np.linalg.norm(PTr_RGB2depth)
			# joints_gt_depth_t = cv2.perspectiveTransform(joints_gt_RGB_t[:, :, :2], PTr_RGB2depth)[0]       # why this operation?
			joints_gt_depth_t = np.array(list(
				map(lambda x: cv2.perspectiveTransform(np.array([x]), PTr_RGB2depth)[0], joints_gt_RGB_t[:, :, :2])))
			joints_gt_depth_t = np.concatenate([joints_gt_depth_t, joints_gt_RGB_t[:, :, 2, None]], axis=2)
			joints_gt_PM_t = np.array(list(
				map(lambda x: cv2.perspectiveTransform(np.array([x]), PTr_RGB)[0], joints_gt_RGB_t[:, :, :2])))
			joints_gt_PM_t = np.concatenate([joints_gt_PM_t, joints_gt_RGB_t[:, :, 2, None]], axis=2)

			pth_cali = os.path.join(dsFd, '{:05d}'.format(i + 1), 'PMcali.npy')
			if not 'simLab' in dsFd:        # only dana has PM data
				caliPM = np.load(pth_cali)
				self.li_caliPM.append(caliPM)       # N x 3 x 45
			# update li
			self.li_joints_gt_RGB.append(joints_gt_RGB_t)  # n_subj x [ n_frm x n_jt x 3]
			self.li_joints_gt_IR.append(joints_gt_IR_t)
			self.li_joints_gt_depth.append(joints_gt_depth_t)
			self.li_joints_gt_PM.append(joints_gt_PM_t)
			self.li_bb_RGB.append(np.array(list(map(ut.get_bbox, joints_gt_RGB_t))))  # n_subj x 45 x4
			self.li_bb_IR.append(np.array(list(map(ut.get_bbox, joints_gt_IR_t))))
			self.li_bb_depth.append(np.array(list(map(ut.get_bbox, joints_gt_depth_t))))
			self.li_bb_PM.append(np.array(list(map(ut.get_bbox, joints_gt_PM_t))))  # only upper level is list
			##  keep the ratio kept bb ( with respect to patch usually square), only provides sq as reader should ind from feeder.
			self.li_bb_sq_RGB.append(np.array(list(map(lambda x: ut.get_bbox(x, rt_xy=1), joints_gt_RGB_t))))  # n_subj x 45 x4
			self.li_bb_sq_IR.append(np.array(list(map(lambda x: ut.get_bbox(x, rt_xy=1), joints_gt_IR_t))))
			self.li_bb_sq_depth.append(np.array(list(map(lambda x: ut.get_bbox(x, rt_xy=1), joints_gt_depth_t))))
			self.li_bb_sq_PM.append(np.array(list(map(lambda x: ut.get_bbox(x, rt_xy=1), joints_gt_PM_t))))  # only upper level is list

		for i in idxs_subj:
			for cov in opts.cov_li:  # add pth Descriptor
				for j in range(n_frm):
					pthDesc_li.append([i + 1, cov, j + 1])  # file idx 1 based,  sample idx 0 based
		self.pthDesc_li = pthDesc_li    # id is 1 started for file compatibility
		self.n_smpl = len(pthDesc_li)       # cov frame subj

	def get_array_joints(self, idx_smpl=0, mod='depthRaw', if_sq_bb=True):
		'''
		index sample function in with flattened order with given modalities. It could be raw form array or image, so we call it array.
		corresponding joints will be returned too.  depth and PM are perspective transformed. bb also returned
		:param idx_smpl: the index number base 0
		:param mod:   4 modality including raw data. PM raw for real pressure data
		:return:
		'''
		id_subj, cov, id_frm = self.pthDesc_li[idx_smpl]    # id for file , base 1
		if_PMreal = False
		if mod == 'PMreal':
			if_PMreal = True
			mod = 'PMarray' # for raw array reading
		arr = getImg_dsPM(dsFd=self.dsFd, idx_subj=id_subj, modality=mod, cov=cov, idx_frm=id_frm)
		if if_PMreal:
			scal_caliPM = self.li_caliPM[id_subj - 1][cov_dict[cov], id_frm - 1]
			arr = arr * scal_caliPM # map to real

		# get unified jt name, get rid of raw extenstion
		if 'depth' in mod:
			mod = 'depth'
		if 'IR' in mod:
			mod = 'IR'
		mod = uni_mod(mod)  # to unify name for shared annotation
		joints_gt = getattr(self, 'li_joints_gt_{}'.format(mod))
		jt = joints_gt[id_subj-1][id_frm-1]
		if if_sq_bb:    # give the s
			bb = getattr(self, 'li_bb_sq_{}'.format(mod))[id_subj - 1][id_frm - 1]
		else:
			bb = getattr(self, 'li_bb_{}'.format(mod))[id_subj-1][id_frm-1]
		return arr, jt, bb

	def get_PTr_A2B(self, idx=0, modA='IR', modB='depthRaw'):
		'''
		get PTr from A2B
		:param idx:
		:param modA:
		:param modB:
		:return:
		'''
		id_subj, cov, id_frm = self.pthDesc_li[idx]  # id for file , base 1
		modA = uni_mod(modA)
		modB = uni_mod(modB)
		PTrA = self.dct_li_PTr[modA][id_subj - 1]  # subj -1 for the li index
		PTrB = self.dct_li_PTr[modB][id_subj - 1]
		PTr_A2B = np.dot(np.linalg.inv(PTrB), PTrA)
		PTr_A2B = PTr_A2B / np.linalg.norm(PTr_A2B)  # normalize

		return PTr_A2B

	def get_array_A2B(self, idx=0, modA='IR', modB='depthRaw'):
		'''
		Get array A after project to B space.
		:param idx:
		:param modA:
		:param modB:
		:return:
		'''
		id_subj, cov, id_frm = self.pthDesc_li[idx]  # id for file , base 1
		arr = getImg_dsPM(dsFd=self.dsFd, idx_subj=id_subj, modality=modA, cov=cov, idx_frm=id_frm) # original A
		PTr_A2B = self.get_PTr_A2B(idx=idx, modA=modA, modB=modB)
		modB = uni_mod(modB)    # the unified name
		# arr_t = transform.warp(arr, PTr_A2B, output_shape=(self.h, self.w), preserve_range=True)
		sz_B = getattr(self, 'sz_{}'.format(modB))
		dst = cv2.warpPerspective(arr, PTr_A2B, tuple(sz_B))
		return dst

	def bb2ori(self, jts, mod='depth'):
		'''
		recover all joints corresponding to idxs to original image space.
		note:, not flexible, get rid later
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

		## all bbs
		bbs = []  # flattened corresponding bbb
		for n_subj, cov, n_frm in self.pthDesc_li:
			bbs.append(bbs_in[n_subj-1][n_frm-1])

		jts_ori = ut.jt_bb2ori(jts, self.sz_pch, bbs)   # better to be outside
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
		jts_cam = ut.pixel2cam(jts_ori, f, c)       # (x, y, depth) from depth map loop the x, y

		return jts_cam, jts_ori

	def get_ptc(self, idx=0, id_bbType=0):
		'''
		get point cloud with idx
		:param idx:
		:param id_bbType: 0 no , bb  1 for pre_calcultated bb, other type not making yet
		:return:
		'''
		arr, jt, bb = self.get_array_joints(idx_smpl=idx)   #df depthRaw
		if 0 == id_bbType:
			bb=None
		elif 0 == id_bbType:
			bb = bb
		else:
			print('type {} not implemented yet'.format(id_bbType))
		ptc = ut.get_ptc(arr, self.f_d, self.c_d, bb)
		return ptc

	def get_phy(self, idx=0):
		'''
		get the physique parameter of idx sample
		:param idx: the index of the sample
		:return:
		'''
		n_subj, cov, n_frm = self.pthDesc_li[idx]
		phyVec = self.phys_arr[n_subj -1]  # 1 index to 0
		return phyVec
