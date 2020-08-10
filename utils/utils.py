"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import re
import sys
import numpy as np
from PIL import Image
import os.path as osp
import cv2
import matplotlib.pyplot as plt
from skimage import io, transform, img_as_ubyte
import math
import random
import time


def diagnose_network(net, name='network'):
	"""Calculate and print the mean of average absolute(gradients)

	Parameters:
		net (torch network) -- Torch network
		name (str) -- the name of the network
	"""
	mean = 0.0
	count = 0
	for param in net.parameters():
		if param.grad is not None:
			mean += torch.mean(torch.abs(param.grad.data))
			count += 1
	if count > 0:
		mean = mean / count
	print(name)
	print(mean)


def save_image(image_numpy, image_path):
	"""Save a numpy image to the disk

	Parameters:
		image_numpy (numpy array) -- input numpy array
		image_path (str)          -- the path of the image
	"""
	image_pil = Image.fromarray(image_numpy)
	image_pil.save(image_path)




def print_numpy(x, val=True, shp=False):
	"""Print the mean, min, max, median, std, and size of a numpy array

	Parameters:
		val (bool) -- if print the values of the numpy array
		shp (bool) -- if print the shape of the numpy array
	"""
	x = x.astype(np.float64)
	if shp:
		print('shape,', x.shape)
	if val:
		x = x.flatten()
		print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
			np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))



def cam2pixel(cam_coord, f, c):
	x = cam_coord[..., 0] / cam_coord[..., 2] * f[0] + c[0]
	y = cam_coord[..., 1] / cam_coord[..., 2] * f[1] + c[1]
	z = cam_coord[..., 2]

	return x, y, z


# def pixel2cam(pixel_coord, f, c):     # old one x,y,z seperated
# 	print('call ut pixel2cam')
# 	pixel_coord = pixel_coord.astype(np.float)
# 	x = (pixel_coord[..., 0] - float(c[0])) / float(f[0]) * pixel_coord[..., 2]
# 	y = (pixel_coord[..., 1] - float(c[1])) / float(f[1]) * pixel_coord[..., 2]
# 	z = pixel_coord[..., 2]
# 	if True:
# 		print('before input', pixel_coord.dtype)
# 		print('get x type', x.dtype)
# 	return x, y, z


def get_bbox(joint_img, rt_margin=1.2, rt_xy=0):
	'''
	get the bounding box from joint gt min max, with a margin ratio.
	:param joint_img:
	:param rt_margin:
	:param rt_xy:   the ratio of x/y . 0 for original bb size. most times 1 for square input patch. Can be gotten from sz_pch[0]/sz_pch[1].
	:return:
	'''
	bb = np.zeros((4))
	xmin = np.min(joint_img[:, 0])
	ymin = np.min(joint_img[:, 1])
	xmax = np.max(joint_img[:, 0])
	ymax = np.max(joint_img[:, 1])
	width = xmax - xmin - 1
	height = ymax - ymin - 1
	if rt_xy:
		c_x = (xmin+xmax)/2.
		c_y = (ymin+ymax)/2.
		aspect_ratio = rt_xy
		w= width
		h = height
		if w > aspect_ratio * h:
			h = w / aspect_ratio
		elif w < aspect_ratio * h:
			w = h * aspect_ratio
		bb[2] = w * rt_margin
		bb[3] = h * rt_margin
		bb[0] = c_x - bb[2] / 2.
		bb[1] = c_y - bb[3] / 2.
	else:
		bb[0] = (xmin + xmax) / 2. - width / 2 * rt_margin
		bb[1] = (ymin + ymax) / 2. - height / 2 * rt_margin
		bb[2] = width * rt_margin
		bb[3] = height * rt_margin

	return bb


def nameToIdx(name_tuple, joints_name):  # test, tp,
	'''
	from reference joints_name, change current name list into index form
	:param name_tuple:  The query tuple like tuple(int,) or tuple(tuple(int, ...), )
	:param joints_name:
	:return:
	'''
	jtNm = joints_name
	if type(name_tuple[0]) == tuple:
		# Transer name_tuple to idx
		return tuple(tuple([jtNm.index(tpl[0]), jtNm.index(tpl[1])]) for tpl in name_tuple)
	else:
		# direct transfer
		return tuple(jtNm.index(tpl) for tpl in name_tuple)

# below from the t3d
def getPCK_3d(p1_err, ref=tuple(range(0,155,5))):
	'''
	single N x n_jt  distance vec
	:param p1_err:
	:param ref:
	:return:
	'''
	# 3d PCK_vec
	pck_v = []
	for th in ref:
		n_valid = np.sum(p1_err<th)
		pck_v.append(float(n_valid)/p1_err.size)
	auc = sum(pck_v)/len(pck_v)
	return pck_v, auc

def li2str(li):
	'''
	transfer the lsit into a string. right now is for int only
	:param li:
	:return:
	'''
	return ''.join([str(e) for e in li])

def getNumInStr(str_in, tp=int):
	'''
	get the number in list transferred as type(indicated)
	:param str_in: the input string
	:return:
	'''
	temp = re.findall(r'\d+', str_in)
	res = list(map(tp, temp))
	return res

def make_folder(folder_name):
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)

def mkdirs(paths):
	"""create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
	if isinstance(paths, list) and not isinstance(paths, str):
		for path in paths:
			make_folder(path)
	else:
		make_folder(paths)


def add_pypath(path):
	if path not in sys.path:
		sys.path.insert(0, path)


def diagnose_network(net, name='network'):
	"""Calculate and print the mean of average absolute(gradients)

	Parameters:
		net (torch network) -- Torch network
		name (str) -- the name of the network
	"""
	mean = 0.0
	count = 0
	for param in net.parameters():
		if param.grad is not None:
			mean += torch.mean(torch.abs(param.grad.data))
			count += 1
	if count > 0:
		mean = mean / count
	print(name)
	print(mean)

def ts2cv2(img_ts, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
	'''
	recover the image from tensor to uint8 cv2 fromat with mean and std. Suppose original in 0~1 format. RGB-BGR, cyx -> yxc
	this version is specific for the  H36M version of mean and std
	:param img_ts:
	:param mean:    inherently define the channel to get
	:param std:
	:return:
	'''

	n_ch = len(mean)
	assert len(mean) == len(std), 'mean ch {} std {} donnot match'.format(len(mean), len(std))
	img_ts = img_ts[:n_ch]

	if not isinstance(img_ts, np.ndarray):  # if tensor transfer it
		tmpimg = img_ts.cpu().detach().numpy()
	else:
		tmpimg = img_ts.copy()
	tmpimg = tmpimg * np.array(std).reshape(n_ch, 1, 1) + np.array(mean).reshape(n_ch, 1, 1)
	tmpimg = tmpimg.astype(np.uint8)
	tmpimg = tmpimg[::-1, :, :]  # BGR
	tmpimg = np.transpose(tmpimg, (1, 2, 0)).copy()  # 2 hwc
	return tmpimg


def draw_gaussian(heatmap, center, sigma):
	'''
	will  affect original image
	:param heatmap:
	:param center:
	:param sigma:
	:return:
	'''
	tmp_size = sigma * 3
	mu_x = int(center[0] + 0.5)     # means the pixel center
	mu_y = int(center[1] + 0.5)
	w, h = heatmap.shape[0], heatmap.shape[1]
	ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)] # h , w coordinate  left up corner
	br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
	if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:  # so ul coord as height and width
		return heatmap
	size = 2 * tmp_size + 1
	x = np.arange(0, size, 1, np.float32)
	y = x[:, np.newaxis]
	x0 = y0 = size // 2
	g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
	g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
	g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
	img_x = max(0, ul[0]), min(br[0], h)
	img_y = max(0, ul[1]), min(br[1], w)
	try:
		heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
			heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
			g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
	except:
		print('center', center)
		print('gx, gy', g_x, g_y)
		print('img_x, img_y', img_x, img_y)
	return heatmap

def get_patch(img, joints, bb, sz_pch, if_keepRt=False):
	'''
	make one
	:param img:
	:param joints:
	:param bb:
	:param sz_pch:
	:param if_keepRt:
	:return: joints_t the new coordinate, with 2: dimensions not change but only first 2 dim cropp and resized
	'''
	if if_keepRt:
		print('not implemented')
		return -1

	else:       # distorted change
		if False:        # debug
			print(img.shape)
			print(bb)       # [190.40034719  56.51957281 120.45727246 351.84747306]
			print(joints)
			print(sz_pch)

		new_Xmin = max(bb[0], 0)
		new_Ymin = max(bb[1], 0)
		new_Xmax = min(bb[0]+bb[2], img.shape[1] - 1)     # column - 1
		new_Ymax = min(bb[1]+bb[3], img.shape[0] - 1)

		imCrop = img.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)]
		# print('new shape', [new_Xmin, new_Ymin, new_Xmax, new_Ymax])
		# print(imCrop.shape)
		imgResize = cv2.resize(imCrop, sz_pch,
		                       interpolation=cv2.INTER_NEAREST)  # crop bb resize,  possible distortion.
		imgResize = np.asarray(imgResize, dtype='float32')  # H*W*C
		# imgResize = (imgResize - Img_mean) / Img_std  # do it outside

		## label        crop scale cood,   depth scale
		# joints_t = np.zeros_like(joints, dtype='float32')
		joints_t = np.copy(joints)
		joints_t[:, 0] = (joints[:, 0] - new_Xmin) * sz_pch[0] / (new_Xmax - new_Xmin)
		joints_t[:, 1] = (joints[:, 1] - new_Ymin) * sz_pch[1] / (new_Ymax - new_Ymin)

		# imageOutputs[:, :, 0] = imgResize
		# imgResize = imgResize.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
		return imgResize, joints_t

def normImg(img, if_toClr=True):
	'''
	normalize image to 0 to 255, make to 3 channels
	:param img:
	:param if_toClr: if convert to color
	:return:
	'''
	v_max = img.max()
	v_min = img.min()
	rst = ((img.astype(float) - v_min) / (v_max-v_min) * 255).astype(np.uint8)
	if if_toClr and img.ndim < 3:   # ad dim to color
		rst = np.stack([rst, rst, rst], axis=2)
	return rst

def jt_bb2ori(jts, sz_pch, bbs):
	'''
	list of jts and bbs to recover back to the list of jts in ori. jts N x n_subj x n_dim(>2).
	bb here is x_ul, y_ul,  w, h
	:param jts:  input joints nx3
	:param sz_pch:
	:param bb:
	:return:  np array N x n_jt x n_dim
	'''

	assert len(jts)<=len(bbs)   # can give full bbs for test
	N = len(jts)
	jts_ori = []
	for i in range(N):
		jts_subj = jts[i]
		bb = bbs[i]
		w = bb[2]
		h = bb[3]
		jts_T = jts_subj.copy()
		jts_T[:,0] = jts_subj[:,0] * w/sz_pch[0] + bb[0]
		jts_T[:,1] = jts_subj[:, 1] * h/sz_pch[1] + bb[1]
		jts_ori.append(jts_T)

	return np.array(jts_ori)

def cam2pixel(cam_coord, f, c):
	x = cam_coord[..., 0] / cam_coord[..., 2] * f[0] + c[0]
	y = cam_coord[..., 1] / cam_coord[..., 2] * f[1] + c[1]
	z = cam_coord[..., 2]

	return x, y, z


def pixel2cam(pixel_coord, f, c):
	pixel_coord = pixel_coord.astype(float)
	jt_cam = np.zeros_like(pixel_coord)
	jt_cam[..., 0] = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
	jt_cam[..., 1] = (pixel_coord[..., 1] - c[1]) / f[1] * pixel_coord[..., 2]
	jt_cam[..., 2] = pixel_coord[..., 2]

	return jt_cam

def get_ptc(depth, f, c, bb=None):
	'''
	get the list of the point cloud in flatten order, row -> column order.
	:param depth: 2d array with real depth value.
	:param f:
	:param c:
	:param bb: if cropping the image and only show the bb area, default none.
	:return: np array of vts list
	'''
	h, w = depth.shape
	vts = []    # lift for rst
	if bb is None:
		rg_r = (0, h)
		rg_c = (0, w)
	else:
		rg_r = (bb[1], bb[1]+bb[3])
		rg_c = (bb[0], bb[0]+bb[2])

	for i in range(rg_r[0], rg_r[1]):
		for j in range(rg_c[0], rg_c[1]):
			vts.append([j, i, depth[i,j]])
	vts = np.array(vts)
	# print('call ut_get ptc')
	vts_cam = pixel2cam(vts, f, c)

	if False:
		print('vts 0 to 5', vts[:5])
		print('after to cam is', vts_cam[:5])
	return vts_cam    # make to np array


def rotate_2d(pt_2d, rot_rad):
	x = pt_2d[0]
	y = pt_2d[1]
	sn, cs = np.sin(rot_rad), np.cos(rot_rad)
	xx = x * cs - y * sn
	yy = x * sn + y * cs
	return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
	# augment size with scale
	src_w = src_width * scale
	src_h = src_height * scale
	src_center = np.array([c_x, c_y], dtype=np.float32)

	# augment rotation
	rot_rad = np.pi * rot / 180
	src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
	src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

	dst_w = dst_width
	dst_h = dst_height
	dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
	dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
	dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

	src = np.zeros((3, 2), dtype=np.float32)
	src[0, :] = src_center
	src[1, :] = src_center + src_downdir
	src[2, :] = src_center + src_rightdir

	dst = np.zeros((3, 2), dtype=np.float32)
	dst[0, :] = dst_center
	dst[1, :] = dst_center + dst_downdir
	dst[2, :] = dst_center + dst_rightdir

	if inv:
		trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
	else:
		trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

	return trans
def generate_patch_image(cvimg, bbox, do_flip, scale, rot, do_occlusion, sz_std=(256, 256)):
	img = cvimg.copy()
	# img_height, img_width, img_channels = img.shape
	img_height, img_width = img.shape[:2]   # first 2

	# synthetic occlusion
	if do_occlusion:
		while True:
			area_min = 0.0
			area_max = 0.7
			synth_area = (random.random() * (area_max - area_min) + area_min) * bbox[2] * bbox[3]

			ratio_min = 0.3
			ratio_max = 1 / 0.3
			synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

			synth_h = math.sqrt(synth_area * synth_ratio)
			synth_w = math.sqrt(synth_area / synth_ratio)
			synth_xmin = random.random() * (bbox[2] - synth_w - 1) + bbox[0]
			synth_ymin = random.random() * (bbox[3] - synth_h - 1) + bbox[1]

			if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < img_width and synth_ymin + synth_h < img_height:
				xmin = int(synth_xmin)
				ymin = int(synth_ymin)
				w = int(synth_w)
				h = int(synth_h)
				img[ymin:ymin + h, xmin:xmin + w, :] = np.random.rand(h, w, 3) * 255
				break

	bb_c_x = float(bbox[0] + 0.5 * bbox[2])
	bb_c_y = float(bbox[1] + 0.5 * bbox[3])
	bb_width = float(bbox[2])
	bb_height = float(bbox[3])

	if do_flip:
		img = img[:, ::-1, :]
		bb_c_x = img_width - bb_c_x - 1

	trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, sz_std[0], sz_std[1], scale,
	                                rot, inv=False)
	img_patch = cv2.warpAffine(img, trans, tuple(sz_std), flags=cv2.INTER_LINEAR)
	if img.ndim>2:  # only transpose color one
		img_patch = img_patch[:, :, ::-1].copy()
	img_patch = img_patch.astype(np.float32)

	return img_patch, trans


def get_aug_config():
	scale_factor = 0.25
	rot_factor = 30
	color_factor = 0.2

	scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
	rot = np.clip(np.random.randn(), -2.0,
	              2.0) * rot_factor if random.random() <= 0.6 else 0
	do_flip = random.random() <= 0.5
	c_up = 1.0 + color_factor
	c_low = 1.0 - color_factor
	color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

	do_occlusion = random.random() <= 0.5

	return scale, rot, do_flip, color_scale, do_occlusion


def warp_coord_to_original(joint_out, bbox, center_cam=[0,0,0], sz_out=[64, 64, 64]):
	'''
	from bb coord back to x,y:pix(ori image) z in meter centered in center_cam.
	:param joint_out:
	:param bbox:  N x 4
	:param center_cam:
	:param sz_out:
	:return:
	'''
	bb_3d_shape = [2000, 2000, 2000]
	# joint_out: output from soft-argmax
	rst = np.zeros_like(joint_out)
	# if len(bbox) == 1620:
	# 	print('bbox shp', bbox.shape)
	# 	print('joints_out shp', joint_out.shape)

	rst[...,0] = joint_out[..., 0] / sz_out[0] * bbox[..., 2, None] + bbox[..., 0, None] # 1620? x 27?
	rst[..., 1] = joint_out[..., 1] / sz_out[1] * bbox[..., 3, None] + bbox[..., 1, None]
	if rst.shape[2] == 3:   # if there is 3rd dim
		rst[..., 2] = (joint_out[..., 2] /sz_out[2] * 2. - 1.) * (bb_3d_shape[0] / 2.) + center_cam[2]      # from 0 - 64 to  -2000 - 2000
	return rst  # same dim as jts input


def trans_point2d(pt_2d, trans):
	src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
	dst_pt = np.dot(trans, src_pt)
	return dst_pt[0:2]

def adj_bb(bb, rt_xy=1):
	'''
	according to ratio x, y, adjust the bb with respect ration (rt), keep longer dim unchanged.
	:param bb:
	:param rt_xy:
	:return:
	'''
	bb_n = bb.copy()
	w = bb[2]
	h = bb[3]
	c_x = bb[0] + w / 2.
	c_y = bb[1] + h / 2.
	aspect_ratio =rt_xy
	if w > aspect_ratio * h:
		h = w / aspect_ratio
	elif w < aspect_ratio * h:
		w = h * aspect_ratio
	bb_n[2] = w
	bb_n[3] = h
	bb_n[0] = c_x - w / 2.
	bb_n[1] = c_y - h / 2.

	return np.array(bb_n)


class Timer(object):
	"""A simple timer."""

	def __init__(self):
		self.total_time = 0.
		self.calls = 0
		self.start_time = 0.
		self.diff = 0.
		self.average_time = 0.
		self.warm_up = 0

	def tic(self):
		# using time.time instead of time.clock because time time.clock
		# does not normalize for multithreading
		self.start_time = time.time()

	def toc(self, average=True):
		self.diff = time.time() - self.start_time
		if self.warm_up < 10:  # no verage time at this moment
			self.warm_up += 1
			return self.diff
		else:
			self.total_time += self.diff
			self.calls += 1
			self.average_time = self.total_time / self.calls

		if average:
			return self.average_time
		else:
			return self.diff

	def reset(self):
		self.total_time = 0.
		self.calls = 0
		self.start_time = 0.
		self.diff = 0.
		self.average_time = 0.
		self.warm_up = 0


class AverageMeter(object):
	"""Computes and stores the average and current value, similar to timer"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count if self.count != 0 else 0

def distNorm(pos_pred_src, pos_gt_src, l_std):
	'''
	claculate the normalized distance  between pos_red_src to pos_gt_src
	:param pos_pred_src: the predict pose  nx 2(3)
	:param pos_gt_src:   the target pose
	:param l_std:
	:return: N x n_jt  normalized dist
	'''
	uv_error = pos_pred_src - pos_gt_src
	uv_err = np.linalg.norm(uv_error, axis=2)   # N x n_jt
	head_size = l_std[..., None]    # add last dim  # N x 1
	return uv_err/ head_size     # get the

def pck(errs, joints_vis, ticks):
	'''
	from the distance, calculate the pck value at each ticks.
	if don't want to use mask, simply set all vlaue of joints_vis to 1.
	:param errs: errors.  better to be normalized.  N x n_jt
	:param joints_vis:  visibility. Give all 1 if you want to count all. N xn_jt
	:param ticks:  the ticks need to be evaluated.
	:return: n_jt x n_ticks
	'''
	joints_vis = joints_vis.squeeze()       # N x 14 ?
	cnts = np.sum(joints_vis, axis=0)    # n_jt
	# print('cnts shape', cnts.shape)
	n_jt = errs.shape[1]
	li_pck = []
	jnt_ratio = cnts / np.sum(cnts).astype(np.float64)
	for i in range(len(ticks)):     # from 0
		pck_t = np.zeros(n_jt+1)       # for last mean
		thr = ticks[i]
		hits = np.sum((errs<=thr)*joints_vis, axis=0) # n_jt       60x14x1?
		# print('hits shape', hits.shape)
		# print('cnts shape', cnts.shape)
		pck_t[:n_jt] = hits/cnts     # n_jt     14 to 60
		pck_t[-1] = np.sum(pck_t[:-1]*jnt_ratio) #
		li_pck.append(pck_t)
	pck_all = np.array(li_pck)*100      # 11 x 14   to  %
	# print('pck all shape', pck_all.shape)
	# print('pck all T shape', pck_all.T.shape)
	return pck_all.T        # n_jt x n_ticks

def prt_rst(rst, titles_c, titles_r, width=10, dg_kpt=1, fn_prt=print):
	'''
	print the result with the given column and row numbers
	:param rst:     the result double list
	:param titles_c:    the column titles
	:param titles_r:    the row titles
	:param width:   the space between result
	:param fn_prt: the function you want to use for print, can be logger.info or print.
	:return:
	'''
	# print(rst)
	# print(titles_r)
	# print(titles_c)
	n_r = len(rst)
	n_c = len(rst[0])
	assert len(titles_c) == n_c, 'need {} get titles c {}'.format(n_c, len(titles_c))
	assert len(titles_r) == n_r, 'need {} get titles r {}'.format(n_r, len(titles_r))
	# row_format = "{:>8}" + "{:>15}" * len(nm_li)
	row_format =("{:>" + "{}".format(width) + "}")* (n_c+1)
	fn_prt(row_format.format("", *titles_c))
	row_format = "{:>" + "{}".format(width) + "}" + \
	             ("{:>" + "{}.{}f".format(width, dg_kpt) + "}") * n_c
	for i in range(n_r):
		rst_t = rst[i]
		title_r_t = titles_r[i]
		fn_prt(row_format.format(title_r_t, *rst_t))
