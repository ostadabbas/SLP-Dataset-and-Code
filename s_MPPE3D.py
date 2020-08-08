'''
For 3DMPPE 3d pose estimation
'''
from data.SLP_RD import SLP_RD
from data.SLP_FD import SLP_FD
import argparse
import utils.vis as vis
import utils.utils as ut
import numpy as np
import opt
import cv2
import skimage
import model.A2J as model
import model.anchor as anchor
import math
import torch
from tqdm import tqdm
import json
from os import path
import os
from model.MPPE3D import  get_pose_net
from torch.nn.parallel.data_parallel import DataParallel



joints_name = (
'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder',
'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
skels = (
(0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3),
(0, 4), (4, 5), (5, 6))

def main():
	opts = opt.parseArgs()
	opts.prep = 'MPPE3D'    # set for specific script   fix those parameters
	opts.if_bb = False  # not using bb, give ori directly
	opts.sz_pch = [256, 256]
	opts.cov_li = ['uncover']     # only uncover
	opts.model = 'MPPE3D'
	opts.trainset = ['H36M']
	opts = opt.aug_opts(opts)       # add necesary opts parameters   to it
	# opt.print_options(opts)
	# model_dir = 'output/ITOP_A2J_exp/model_dump/ITOP_side_30.pth'  # hardwired due in this test case
	model_path = '/scratch/liu.shu/codesPool/3DMPPE_POSENET_RELEASE/output/model_dump/proto2/snapshot_24.pth.tar'  # hardwired from other repo
	nm_test= 'ori'  # not patch used
	# provided by h36m
	pixel_mean = np.array((0.485, 0.456, 0.406))
	pixel_std = np.array((0.229, 0.224, 0.225))
	sz_out = (64, 64, 64)   # the hm output
	n_jt = 18   #for the h36m version, last thorax is not for test not for drawing, bias not correct, seems to have 19?
	rg = [[0, sz_out[0]], [0, sz_out[1]], [0, sz_out[2]]]    # the depth dim

	SLP_rd = SLP_RD(opts, phase='all')  # all test result
	SLP_fd = SLP_FD(SLP_rd, opts)
	print('processing total subj {}'.format(SLP_rd.n_subj))

	test_dataloaders = torch.utils.data.DataLoader(SLP_fd, batch_size=opts.batch_size,
	                                               shuffle=False, num_workers=0)
	dl_iter = iter(test_dataloaders)
	itr_per_epoch = math.ceil(
		SLP_fd.__len__() / opts.num_gpus / opts.batch_size)  # single ds test on batch share
	if opts.testIter > 0:
		itr_per_epoch = min(itr_per_epoch, opts.testIter)

	# model in
	assert os.path.exists(model_path), 'Cannot find model at ' + model_path

	model = get_pose_net(False, n_jt)
	model = DataParallel(model).cuda()
	ckpt = torch.load(model_path)
	model.load_state_dict(ckpt['network'])
	model.eval()

	test_dataloaders = torch.utils.data.DataLoader(SLP_fd, batch_size=opts.batch_size, shuffle=False, num_workers=0)
	dl_iter = iter(test_dataloaders)
	itr_per_epoch = math.ceil(
		SLP_fd.__len__() / opts.num_gpus / opts.batch_size)  # single ds test on batch share
	if opts.testIter > 0:
		itr_per_epoch = min(itr_per_epoch, opts.testIter)
	preds = []
	bbs = []
	# itr_per_epoch=2
	with torch.no_grad():
		for i in tqdm(range(itr_per_epoch)):
			inp = next(dl_iter)
			input_img = inp['arr_tch']
			bb_tch = inp['bb_tch']
			if False:    # check the shape
				print('image shape', input_img.size())

			# forward
			coord_out = model(input_img)
			# no flipping thing
			coord_out = coord_out.cpu().numpy()
			bb = bb_tch.cpu().numpy()
			if False:    # show the image for checking
				sv_dir = opts.vis_test_dir  # exp/vis/Human36M
				img_patch_vis = ut.ts2cv2(input_img[0])
				idx_test = i * opts.batch_size
				# skels_idx = opts.ref_skels_idx
				skels_idx = skels       # the h36m version
				# get pred2d_patch
				# pred2d_patch = np.zeros((3, n_jt))  # 3xn_jt format
				pred2d_patch = np.zeros((n_jt, 3))  # 3xn_jt format
				# pred2d_patch[:2, :] = coord_out[0, :, :2].transpose(1, 0) / sz_out[0] * opts.sz_pch[0]  # 3 * n_jt  set depth 1 , from coord_out 0 !!
				pred2d_patch[:, :2] = coord_out[0, :, :2]/ sz_out[0] * opts.sz_pch[0]  # no transpose  for new vis

				pred2d_patch[:, 2] = 1      # all vis

				# ut.save_2d_tg3d(img_patch_vis, pred2d_patch, skels_idx, sv_dir, idx=idx_test)  # make sub dir if needed, recover to test set index by indexing.
				img_2d = vis.vis_keypoints(img_patch_vis, pred2d_patch, skels)
				cv2.imshow('2d patch', img_2d)
				cv2.waitKey(0)
				cv2.destroyAllWindows()

				# test 2d in original
				coord_T = coord_out[0]
				bb_T = bb[0]
				coord_ori = ut.warp_coord_to_original(coord_T, bb_T)
				coord_2d_ori = coord_ori.copy()
				coord_2d_ori[:,2] = 1
				arr_rd, jt_rd, bb_rd = SLP_rd.get_array_joints(idx_test, mod='RGB') # get original image
				img_2d_ori = vis.vis_keypoints(arr_rd, coord_2d_ori, skels)
				cv2.imshow('2d ori', img_2d_ori)
				cv2.waitKey(0)
				cv2.destroyAllWindows()

				# for 3d
				# ut.save_3d_tg3d(coord_out[0], sv_dir, skels_idx, idx=idx_test,
				vis.vis_3d(coord_out[0],skels, rg=rg)
			preds.append(coord_out)
			bbs.append(bb)
	# evaluate
	preds = np.concatenate(preds, axis=0)       # in hm space
	# print('before concate bb[0] is', bb[0])
	bbs = np.concatenate(bbs, axis=0)   # N x
	bbs = bbs.squeeze()
	# print('after concate, bb is', bbs.shape)
	print('total result shape', preds.shape)

	# get 2d in ori
	center_cam = [0, 0, 0]
	sz_out = [64, 64, 64]
	bb_3d_shape = [2000, 2000, 2000]
	preds_ori = ut.jt_bb2ori(preds, sz_out, bbs)
	preds_ori[..., 2] = (preds_ori[..., 2] / sz_out[2] * 2. - 1.) * (bb_3d_shape[0] / 2.) + center_cam[2]
	if False:        # test a single image
		idx_test = 20
		arr_rd, jt_rd, bb_rd = SLP_rd.get_array_joints(idx_test, mod='RGB')  # get original image
		print('preds in patch', preds[idx_test])
		print('recovered to ori', preds_ori[idx_test])

		preds_ori_T = preds_ori[idx_test].copy()
		preds_ori_T[:,2]=1
		img_2d_ori = vis.vis_keypoints(arr_rd, preds_ori_T, skels)
		cv2.imshow('read in', img_2d_ori)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	ut.make_folder(opts.rst_dir)
	svPth = path.join(opts.rst_dir, 'SLP_test_3d_{}.json'.format(nm_test))  # add suffix
	print('saved to {}'.format(svPth))
	sv_preds={'joints_name':joints_name, 'preds_3d_hm':preds.tolist(), 'preds_ori':preds_ori.tolist(), 'bbs':bbs.tolist()}

	with open(svPth, 'w') as f:
		json.dump(sv_preds, f, allow_nan=True)
	# sv for later
	# tester._evaluate(preds, cfg.result_dir)

if __name__ == "__main__":
	main()


