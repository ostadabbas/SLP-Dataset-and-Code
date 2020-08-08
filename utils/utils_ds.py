'''
ds related tools
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random
import time
import torch
import copy
import math
from torch.utils.data.dataset import Dataset
from utils.vis import vis_keypoints
# from config import cfg
from math import floor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import math
import numpy as np

from utils.transforms import transform_preds

def get_aug_config():
	scale_factor = 0.25
	rot_factor = 30
	color_factor = 0.2

	scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
	rot = np.clip(np.random.randn(), -2.0,
	              2.0) * rot_factor if random.random() <= 0.6 else 0        # -60 to 60
	do_flip = random.random() <= 0.5
	c_up = 1.0 + color_factor
	c_low = 1.0 - color_factor
	color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

	do_occlusion = random.random() <= 0.5

	return scale, rot, do_flip, color_scale, do_occlusion


def generate_patch_image(cvimg, bbox, do_flip, scale, rot, do_occlusion, input_shape=(256, 256)):
	# return skimage RGB,  h,w,c
	# flip first , then trans ( rot, scale,
	img = cvimg.copy()
	img_height, img_width, img_channels = img.shape  # h,w,c        # too many to unpack?

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
				img[ymin:ymin + h, xmin:xmin + w, :] = np.random.rand(h, w, img_channels) * 255
				break

	bb_c_x = float(bbox[0] + 0.5 * bbox[2])
	bb_c_y = float(bbox[1] + 0.5 * bbox[3])
	bb_width = float(bbox[2])
	bb_height = float(bbox[3])

	if do_flip:
		img = img[:, ::-1, :]
		bb_c_x = img_width - bb_c_x - 1

	trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale, rot,
	                                inv=False)  # is bb aspect needed? yes, otherwise patch distorted
	img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])),
	                           flags=cv2.INTER_LINEAR)  # is there channel requirements
	# if len(img_patch.shape)==3:     #  I don't think it is needed as original is already single channel
	# 	img_patch = img_patch[:, :, ::-1].copy()

	img_patch = img_patch.copy().astype(np.float32)

	return img_patch, trans


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


def trans_point2d(pt_2d, trans):
	src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
	dst_pt = np.dot(trans, src_pt)
	return dst_pt[0:2]


def generate_target(joints, joints_vis, sz_hm=[64, 64], sigma=2, gType='gaussian'):
	'''
	:param joints:  [num_joints, 3]
	:param joints_vis: n_jt vec     #  original n_jt x 3
	:param sigma: for gaussian gen, 3 sigma rule for effective area.  hrnet default 2.
	:return: target, target_weight(1: visible, 0: invisible),  n_jt x 1
	history: gen directly at the jt position, stride should be handled outside
	'''
	n_jt = len(joints)  #
	target_weight = np.ones((n_jt, 1), dtype=np.float32)
	# target_weight[:, 0] = joints_vis[:, 0]
	target_weight[:, 0] = joints_vis        # wt equals to vis

	assert gType == 'gaussian', \
		'Only support gaussian map now!'

	if gType == 'gaussian':
		target = np.zeros((n_jt,
		                   sz_hm[1],
		                   sz_hm[0]),
		                  dtype=np.float32)

		tmp_size = sigma * 3

		for joint_id in range(n_jt):
			# feat_stride = self.image_size / sz_hm
			# mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
			# mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
			mu_x = int(joints[joint_id][0] + 0.5)   # in hm joints could be in middle,  0.5 to biased to the position.
			mu_y = int(joints[joint_id][1] + 0.5)

			# Check that any part of the gaussian is in-bounds
			ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
			br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
			if ul[0] >= sz_hm[0] or ul[1] >= sz_hm[1] \
					or br[0] < 0 or br[1] < 0:
				# If not, just return the image as is
				target_weight[joint_id] = 0
				continue

			# # Generate gaussian
			size = 2 * tmp_size + 1
			x = np.arange(0, size, 1, np.float32)
			y = x[:, np.newaxis]
			x0 = y0 = size // 2
			# The gaussian is not normalized, we want the center value to equal 1
			g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

			# Usable gaussian range
			g_x = max(0, -ul[0]), min(br[0], sz_hm[0]) - ul[0]
			g_y = max(0, -ul[1]), min(br[1], sz_hm[1]) - ul[1]
			# Image range
			img_x = max(0, ul[0]), min(br[0], sz_hm[0])
			img_y = max(0, ul[1]), min(br[1], sz_hm[1])

			v = target_weight[joint_id]
			if v > 0.5:
				target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
					g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

	# print('min max', target.min(), target.max())
	# if self.use_different_joints_weight:
	# 	target_weight = np.multiply(target_weight, self.joints_weight)

	return target, target_weight


def get_max_preds(batch_heatmaps):
	'''
	get predictions from score maps
	heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
	:return preds [N x n_jt x 2]
	'''
	assert isinstance(batch_heatmaps, np.ndarray), \
		'batch_heatmaps should be numpy.ndarray'
	assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

	batch_size = batch_heatmaps.shape[0]
	num_joints = batch_heatmaps.shape[1]
	width = batch_heatmaps.shape[3]
	heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
	idx = np.argmax(heatmaps_reshaped, 2)
	maxvals = np.amax(heatmaps_reshaped, 2)  # amax, array max

	maxvals = maxvals.reshape((batch_size, num_joints, 1))
	idx = idx.reshape((batch_size, num_joints, 1))

	preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

	preds[:, :, 0] = (preds[:, :, 0]) % width
	preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

	pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
	pred_mask = pred_mask.astype(np.float32)  # clean up if too low confidence (maxvals)

	preds *= pred_mask
	return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
	coords, maxvals = get_max_preds(batch_heatmaps)

	heatmap_height = batch_heatmaps.shape[2]
	heatmap_width = batch_heatmaps.shape[3]

	# post-processing
	if config.TEST.POST_PROCESS:
		for n in range(coords.shape[0]):
			for p in range(coords.shape[1]):
				hm = batch_heatmaps[n][p]
				px = int(math.floor(coords[n][p][0] + 0.5))
				py = int(math.floor(coords[n][p][1] + 0.5))
				if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
					diff = np.array(
						[
							hm[py][px + 1] - hm[py][px - 1],
							hm[py + 1][px] - hm[py - 1][px]
						]
					)
					coords[n][p] += np.sign(diff) * .25

	preds = coords.copy()

	# Transform back
	for i in range(coords.shape[0]):
		preds[i] = transform_preds(
			coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
		)

	return preds, maxvals


def calc_dists(preds, target, normalize):
	# normalized distance on x, y seperately,  anyway
	preds = preds.astype(np.float32)
	target = target.astype(np.float32)
	dists = np.zeros((preds.shape[1], preds.shape[0]))
	for n in range(preds.shape[0]):
		for c in range(preds.shape[1]):
			if target[n, c, 0] > 1 and target[n, c, 1] > 1:
				normed_preds = preds[n, c, :] / normalize[n]
				normed_targets = target[n, c, :] / normalize[n]
				dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
			else:
				dists[c, n] = -1
	return dists

def dist_acc(dists, thr=0.5):
	''' Return percentage below threshold while ignoring values with a -1
	dist has already been normalized
	normalized is simply based on (h,w)/10 std
	'''
	dist_cal = np.not_equal(dists, -1)
	num_dist_cal = dist_cal.sum()
	if num_dist_cal > 0:
		return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
	else:
		return -1

def accuracy(output, target, hm_type='gaussian', thr=0.5):
	'''
	Calculate accuracy according to PCK,
	but uses ground truth heatmap rather than x,y locations
	First value to be returned is average accuracy across 'idxs',
	followed by individual accuracies
	'''
	idx = list(range(output.shape[1]))  # N x n_jts
	norm = 1.0  # norm is fixed ? why ?
	if hm_type == 'gaussian':
		pred, _ = get_max_preds(output)
		target, _ = get_max_preds(target)
		h = output.shape[2]
		w = output.shape[3]
		norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10  # use 0.1 as norm
	dists = calc_dists(pred, target, norm)  # given one , so not normalized

	acc = np.zeros((len(idx) + 1))
	avg_acc = 0
	cnt = 0

	for i in range(len(idx)):
		acc[i + 1] = dist_acc(dists[idx[i]])  # use thr to do it.
		if acc[i + 1] >= 0:
			avg_acc = avg_acc + acc[i + 1]
			cnt += 1

	avg_acc = avg_acc / cnt if cnt != 0 else 0  # cnt how many jts
	if cnt != 0:
		acc[0] = avg_acc
	return acc, avg_acc, cnt, pred


def flip_back(output_flipped, matched_parts):
	'''
	ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
	'''
	assert output_flipped.ndim == 4, \
		'output_flipped should be [batch_size, num_joints, height, width]'

	output_flipped = output_flipped[:, :, :, ::-1]  # mirror

	for pair in matched_parts:      # change channel order
		tmp = output_flipped[:, pair[0], :, :].copy()
		output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
		output_flipped[:, pair[1], :, :] = tmp

	return output_flipped
