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


def pixel2cam(pixel_coord, f, c):
	x = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
	y = (pixel_coord[..., 1] - c[1]) / f[1] * pixel_coord[..., 2]
	z = pixel_coord[..., 2]

	return x, y, z


def get_bbox(joint_img, rt_margin=1.4):
	# bbox extract from keypoint coordinates, ul x,y ,w, h
	bbox = np.zeros((4))
	xmin = np.min(joint_img[:, 0])
	ymin = np.min(joint_img[:, 1])
	xmax = np.max(joint_img[:, 0])
	ymax = np.max(joint_img[:, 1])
	width = xmax - xmin - 1
	height = ymax - ymin - 1

	bbox[0] = (xmin + xmax) / 2. - width / 2 * 1.2
	bbox[1] = (ymin + ymax) / 2. - height / 2 * 1.2
	bbox[2] = width * rt_margin
	bbox[3] = height * rt_margin

	return bbox


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


def tensor2im(input_image, imtype=np.uint8, clipMod='clip01'):
	""""Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
	if not isinstance(input_image, np.ndarray):
		if isinstance(input_image, torch.Tensor):  # get the data from a variable
			image_tensor = input_image.data
		else:
			return input_image
		image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
		if image_numpy.shape[0] == 1:  # grayscale to RGB
			image_numpy = np.tile(image_numpy, (3, 1, 1))
		if 'clip11' == clipMod:
			image_numpy = (np.transpose(image_numpy,
			                            (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
		else:  # for clip 11 operation
			image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * 255.0)  # 01 scale directly
	else:  # if it is a numpy array, do nothing
		image_numpy = input_image
	return image_numpy.astype(imtype)


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
	:param img_ts:
	:param mean:
	:param std:
	:return:
	'''
	if not isinstance(img_ts, np.ndarray):  # if tensor transfer it
		tmpimg = img_ts.cpu().detach().numpy()
	else:
		tmpimg = img_ts.copy()
	tmpimg = tmpimg * np.array(std).reshape(3, 1, 1) + np.array(mean).reshape(3, 1, 1)
	tmpimg = tmpimg.astype(np.uint8)
	tmpimg = tmpimg[::-1, :, :]  # BGR
	tmpimg = np.transpose(tmpimg, (1, 2, 0)).copy()  # x, y , c
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
	mu_x = int(center[0] + 0.5)
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
	jt_cam = np.zeros_like(pixel_coord)
	jt_cam[..., 0] = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
	jt_cam[..., 1] = (pixel_coord[..., 1] - c[1]) / f[1] * pixel_coord[..., 2]
	jt_cam[..., 2] = pixel_coord[..., 2]

	return jt_cam
