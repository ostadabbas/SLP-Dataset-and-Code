'''
2d pose estimation handling
'''

from data.SLP_RD import SLP_RD
from data.SLP_FD import SLP_FD
import utils.vis as vis
import utils.utils as ut
import numpy as np
import opt
import cv2
import torch
import json
from os import path
import os
from utils.logger import Colorlogger
from utils.utils_tch import get_model_summary
from core.loss import JointsMSELoss
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
from utils.utils_ds import accuracy, flip_back
from utils.visualizer import Visualizer

# opts outside?
opts = opt.parseArgs()
if 'depth' in opts.mod_src[0]:  # the leading modalities, only depth use tight bb other raw image size
	opts.if_bb = True  # not using bb, give ori directly
else:
	opts.if_bb = False  #
exec('from model.{} import get_pose_net'.format(opts.model))  # pose net in
opts = opt.aug_opts(opts)  # add necesary opts parameters   to it
# opt.print_options(opts)

def train(loader, ds_rd, model, criterion, optimizer, epoch, n_iter=-1, logger=None, opts=None, visualizer=None):
	'''
	iter through epoch , return rst{'acc', loss'} each as list can be used outside for updating.
	:param loader:
	:param model:
	:param criterion:
	:param optimizer:
	:param epoch:  for print infor
	:param n_iter: the iteration wanted, -1 for all iters
	:param opts: keep some additional controls
	:param visualizer: for visualizer
	:return:
	'''
	batch_time = ut.AverageMeter()
	data_time = ut.AverageMeter()
	losses = ut.AverageMeter()
	acc = ut.AverageMeter()

	# switch to train mode
	model.train()
	end = time.time()
	li_loss = []
	li_acc = []
	for i, inp_dct in enumerate(loader):
		# get items
		if i>=n_iter and n_iter>0:    # break if iter is set and i is greater than that
			break
		input = inp_dct['pch']
		target = inp_dct['hms']     # 14 x 64 x 1??
		target_weight = inp_dct['joints_vis']

		# measure data loading time     weight, visible or not
		data_time.update(time.time() - end)

		# compute output
		outputs = model(input)      # no need to cuda it?

		target = target.cuda(non_blocking=True)
		target_weight = target_weight.cuda(non_blocking=True)

		if isinstance(outputs, list):       # list multiple stage version
			loss = criterion(outputs[0], target, target_weight)
			for output in outputs[1:]:
				loss += criterion(output, target, target_weight)
		else:
			output = outputs
			loss = criterion(output, target, target_weight)


		# compute gradient and do update step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure accuracy and record loss
		losses.update(loss.item(), input.size(0))
		_, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
		                                 target.detach().cpu().numpy())  # hm directly, with normalize with 1/10 dim,  pck0.5,  cnt: n_smp,  pred
		acc.update(avg_acc, cnt)  # keep average acc

		if visualizer and 0 == i % opts.update_html_freq:     # update current result, get vis dict
			n_jt = ds_rd.joint_num_ori
			mod0 = opts.mod_src[0]
			mean = ds_rd.means[mod0]
			std = ds_rd.stds[mod0]
			img_patch_vis = ut.ts2cv2(input[0], mean, std)  # to CV BGR, mean std control channel detach inside
			# pseudo change
			cm = getattr(cv2, ds_rd.dct_clrMap[mod0])
			img_patch_vis = cv2.applyColorMap(img_patch_vis, cm)[...,::-1]  # RGB

			# get pred
			pred2d_patch = np.ones((n_jt, 3))  # 3rd for  vis
			pred2d_patch[:, :2] = pred[0] / opts.out_shp[0] * opts.sz_pch[1]
			img_skel = vis.vis_keypoints(img_patch_vis, pred2d_patch, ds_rd.skels_idx)

			hm_gt = target[0].cpu().detach().numpy().sum(axis=0)    # HXW
			hm_gt = ut.normImg(hm_gt)

			hm_pred = output[0].detach().cpu().numpy().sum(axis=0)
			hm_pred = ut.normImg(hm_pred)
			img_cb = vis.hconcat_resize([img_skel, hm_gt, hm_pred])
			vis_dict = {'img_cb': img_cb}
			visualizer.display_current_results(vis_dict, epoch, False)

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % opts.print_freq == 0:
			msg = 'Epoch: [{0}][{1}/{2}]\t' \
			      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
			      'Speed {speed:.1f} samples/s\t' \
			      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
			      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
			      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
				epoch, i, len(loader), batch_time=batch_time,
				speed=input.size(0) / batch_time.val,
				data_time=data_time, loss=losses, acc=acc)
			logger.info(msg)
			li_loss.append(losses.val)   # the current loss
			li_acc.append(acc.val)

	return {'losses':li_loss, 'accs':li_acc}


def validate(loader, ds_rd, model, criterion, n_iter=-1, logger=None, opts=None, if_svVis=False, visualizer=None):
	'''
	loop through loder, all res, get preds and gts and normled dist.
	With flip test for higher acc.
	for preds, bbs, jts_ori, jts_weigth out, recover preds_ori, dists_nmd, pckh( dist and joints_vis filter, , print, if_sv then save all these
	:param loader:
	:param ds_rd: the reader, givens the length and flip pairs
	:param model:
	:param criterion:
	:param optimizer:
	:param epoch:
	:param n_iter:
	:param logger:
	:param opts:
	:return:
	'''
	batch_time = ut.AverageMeter()
	losses = ut.AverageMeter()
	acc = ut.AverageMeter()

	# switch to evaluate mode
	model.eval()

	num_samples = ds_rd.n_smpl
	n_jt = ds_rd.joint_num_ori

	# to accum rst
	preds_hm = []
	bbs = []
	li_joints_ori = []
	li_joints_vis = []
	li_l_std_ori = []
	with torch.no_grad():
		end = time.time()
		for i, inp_dct in enumerate(loader):
			# compute output
			input = inp_dct['pch']
			target = inp_dct['hms']
			target_weight = inp_dct['joints_vis']
			bb = inp_dct['bb']
			joints_ori = inp_dct['joints_ori']
			l_std_ori = inp_dct['l_std_ori']
			if i>= n_iter and n_iter>0:     # limiting iters
				break
			outputs = model(input)
			if isinstance(outputs, list):
				output = outputs[-1]
			else:
				output = outputs
			output_ori = output.clone()     # original output of original image
			if opts.if_flipTest:
				input_flipped = input.flip(3).clone()       # flipped input
				outputs_flipped = model(input_flipped)      # flipped output
				if isinstance(outputs_flipped, list):
					output_flipped = outputs_flipped[-1]
				else:
					output_flipped = outputs_flipped
				output_flipped_ori = output_flipped.clone() # hm only head changed? not possible??
				output_flipped = flip_back(output_flipped.cpu().numpy(),
				                           ds_rd.flip_pairs)
				output_flipped = torch.from_numpy(output_flipped.copy()).cuda() # N x n_jt xh x w tch

				# feature is not aligned, shift flipped heatmap for higher accuracy
				if_shiftHM = True  # no idea why
				if if_shiftHM:      # check original
					# print('run shift flip')
					output_flipped[:, :, :, 1:] = \
						output_flipped.clone()[:, :, :, 0:-1]

				output = (output + output_flipped) * 0.5

			target = target.cuda(non_blocking=True)
			target_weight = target_weight.cuda(non_blocking=True)
			loss = criterion(output, target, target_weight)

			num_images = input.size(0)
			# measure accuracy and record loss
			losses.update(loss.item(), num_images)
			_, avg_acc, cnt, pred_hm = accuracy(output.cpu().numpy(),
			                                 target.cpu().numpy())
			acc.update(avg_acc, cnt)

			# preds can be furhter refined with subpixel trick, but it is already good enough.
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			# keep rst
			preds_hm.append(pred_hm)        # already numpy, 2D
			bbs.append(bb.numpy())
			li_joints_ori.append(joints_ori.numpy())
			li_joints_vis.append(target_weight.cpu().numpy())
			li_l_std_ori.append(l_std_ori.numpy())

			if if_svVis and 0 == i % opts.svVis_step:
				sv_dir = opts.vis_test_dir  # exp/vis/Human36M
				# batch version
				mod0 = opts.mod_src[0]
				mean = ds_rd.means[mod0]
				std = ds_rd.stds[mod0]
				img_patch_vis = ut.ts2cv2(input[0], mean, std) # to CV BGR
				img_patch_vis_flipped = ut.ts2cv2(input_flipped[0], mean, std) # to CV BGR
				# pseudo change
				cm = getattr(cv2,ds_rd.dct_clrMap[mod0])
				img_patch_vis = cv2.applyColorMap(img_patch_vis, cm)
				img_patch_vis_flipped = cv2.applyColorMap(img_patch_vis_flipped, cm)

				# original version get img from the ds_rd , different size , plot ing will vary from each other
				# warp preds to ori
				# draw and save  with index.

				idx_test = i * opts.batch_size  # image index
				skels_idx = ds_rd.skels_idx
				# get pred2d_patch
				pred2d_patch = np.ones((n_jt, 3))  # 3rd for  vis
				pred2d_patch[:,:2] = pred_hm[0] / opts.out_shp[0] * opts.sz_pch[1]      # only first
				vis.save_2d_skels(img_patch_vis, pred2d_patch, skels_idx, sv_dir, suffix='-'+mod0,
				                  idx=idx_test)  # make sub dir if needed, recover to test set index by indexing.
				# save the hm images. save flip test
				hm_ori = ut.normImg(output_ori[0].cpu().numpy().sum(axis=0))    # rgb one
				hm_flip = ut.normImg(output_flipped[0].cpu().numpy().sum(axis=0))
				hm_flip_ori = ut.normImg(output_flipped_ori[0].cpu().numpy().sum(axis=0))
				# subFd = mod0+'_hmFlip_ori'
				# vis.save_img(hm_flip_ori, sv_dir, idx_test, sub=subFd)

				# combined
				# img_cb = vis.hconcat_resize([img_patch_vis, hm_ori, img_patch_vis_flipped, hm_flip_ori])        # flipped hm
				# subFd = mod0+'_cbFlip'
				# vis.save_img(img_cb, sv_dir, idx_test, sub=subFd)


			if i % opts.print_freq == 0:
				msg = 'Test: [{0}/{1}]\t' \
				      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
				      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
				      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					i, len(loader), batch_time=batch_time,
					loss=losses, acc=acc)
				logger.info(msg)

	preds_hm = np.concatenate(preds_hm,axis=0)      # N x n_jt  x 2
	bbs = np.concatenate(bbs, axis=0)
	joints_ori = np.concatenate(li_joints_ori, axis=0)
	joints_vis = np.concatenate(li_joints_vis, axis=0)
	l_std_ori_all = np.concatenate(li_l_std_ori, axis=0)

	preds_ori = ut.warp_coord_to_original(preds_hm, bbs, sz_out=opts.out_shp)
	err_nmd = ut.distNorm(preds_ori,  joints_ori, l_std_ori_all)
	ticks = np.linspace(0,0.5,11)   # 11 ticks
	pck_all = ut.pck(err_nmd, joints_vis, ticks=ticks)

	# save to plain format for easy processing
	rst = {
		'preds_ori':preds_ori.tolist(),
		'joints_ori':joints_ori.tolist(),
		'l_std_ori_all': l_std_ori_all.tolist(),
		'err_nmd': err_nmd.tolist(),
		'pck': pck_all.tolist()
	}

	return rst


def main():
	# get logger
	if_test = opts.if_test
	if if_test:
		log_suffix = 'test'
	else:
		log_suffix = 'train'
	logger = Colorlogger(opts.log_dir, '{}_logs.txt'.format(log_suffix))    # avoid overwritting, will append
	opt.set_env(opts)
	opt.print_options(opts, if_sv=True)
	n_jt = SLP_RD.joint_num_ori     #

	# get model
	model = get_pose_net(in_ch=opts.input_nc, out_ch=n_jt)      # why call it get c

	# define loss function (criterion) and optimizer
	criterion = JointsMSELoss(      # try to not use weights
		use_target_weight=True
	).cuda()

	# ds adaptor
	SLP_rd_train = SLP_RD(opts, phase='train')  # all test result
	SLP_fd_train = SLP_FD(SLP_rd_train, opts, phase='train', if_sq_bb=True)
	train_loader = DataLoader(dataset=SLP_fd_train, batch_size= opts.batch_size // len(opts.trainset),
	                    shuffle=True, num_workers=opts.n_thread, pin_memory=opts.if_pinMem)

	SLP_rd_test = SLP_RD(opts, phase=opts.test_par)  # all test result      # can test against all controled in opt
	SLP_fd_test = SLP_FD(SLP_rd_test,  opts, phase='test', if_sq_bb=True)
	test_loader = DataLoader(dataset=SLP_fd_test, batch_size = opts.batch_size // len(opts.trainset),
	                          shuffle=False, num_workers=opts.n_thread, pin_memory=opts.if_pinMem)

	# for visualzier
	if opts.display_id > 0:
		visualizer = Visualizer(opts)  # only plot losses here, a loss log comes with it,
	else:
		visualizer = None
	# get optmizer
	best_perf = 0.0
	last_epoch = -1
	optimizer = Adam(model.parameters(), lr=opts.lr)
	checkpoint_file = os.path.join(
		opts.model_dir, 'checkpoint.pth')
	if 0 == opts.start_epoch or not path.exists(checkpoint_file):  #    from scratch
		begin_epoch =  0     # either set or not exist all the same from 0
		losses = []     # for tracking model performance.
		accs= []
	else:  # get chk points
		logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
		checkpoint = torch.load(checkpoint_file)
		begin_epoch = checkpoint['epoch']
		best_perf = checkpoint['perf']
		last_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])  # here should be cuda setting
		losses = checkpoint['losses']
		accs = checkpoint['accs']

		optimizer.load_state_dict(checkpoint['optimizer'])
		logger.info("=> loaded checkpoint '{}' (epoch {})".format(
			checkpoint_file, checkpoint['epoch']))

	milestones = opts.lr_dec_epoch
	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
		optimizer, milestones, opts.lr_dec_factor,
		last_epoch=last_epoch
	)  # scheduler will be set to place given last from checkpoints
	if opts.epoch_step > 0:
		end_epoch = min(opts.end_epoch, opts.start_epoch + opts.epoch_step)
	else:
		end_epoch = opts.end_epoch

	dump_input = torch.rand(
		(1, opts.input_nc, opts.sz_pch[1], opts.sz_pch[0])
	)

	logger.info(get_model_summary(model, dump_input))
	model = torch.nn.DataParallel(model, device_ids=opts.gpu_ids).cuda()

	n_iter = opts.trainIter  # only for test purpose     quick test
	if not if_test:
		for epoch in range(begin_epoch,end_epoch):
			if opts.display_id > 0:
				visualizer.reset()      # clean up the vis
			# train for one epoch
			rst_trn = train(train_loader, SLP_rd_train, model, criterion, optimizer, epoch, n_iter=n_iter, logger=logger, opts=opts, visualizer=visualizer)
			losses += rst_trn['losses']
			accs += rst_trn['accs']

			# evaluate on validation set    to update
			rst_test = validate(
				test_loader, SLP_rd_test, model, criterion,
				n_iter=n_iter, logger=logger, opts=opts)   # save preds, gt, preds_in ori, idst_normed to recovery, error here for last epoch?
			pck_all = rst_test['pck']
			perf_indicator = pck_all[-1][-1] # the last entry
			pckh05 = np.array(pck_all)[:, -1]   # the last indicies     15 x 11 last
			titles_c = list(SLP_rd_test.joints_name[:SLP_rd_test.joint_num_ori]) + ['total']
			ut.prt_rst([pckh05], titles_c, ['pckh0.5'], fn_prt=logger.info)

			lr_scheduler.step()     # new version updating here
			if perf_indicator >= best_perf:
				best_perf = perf_indicator
				best_model = True
			else:
				best_model = False

			logger.info('=> saving checkpoint to {}'.format(opts.model_dir))
			ckp = {
				'epoch': epoch + 1,     # epoch to next, after finish 0 this is 1
				'model': opts.model,
				'state_dict': model.module.state_dict(),
				'best_state_dict': model.module.state_dict(),
				'perf': perf_indicator,
				'optimizer': optimizer.state_dict(),
				'losses': losses,       # for later updating
				'accs': accs,
			}
			torch.save(ckp, os.path.join(opts.model_dir, 'checkpoint.pth'))
			if best_model:
				torch.save(ckp, os.path.join(opts.model_dir, 'model_best.pth'))
			# save directly, if statebest save another

		final_model_state_file = os.path.join(
			opts.model_dir, 'final_state.pth'     # only after last iters
		)
		logger.info('=> saving final model state to {}'.format(
			final_model_state_file)
		)
		torch.save(model.module.state_dict(), final_model_state_file)

	# single test with loaded model, save the result
	logger.info('----run final test----')
	rst_test = validate(
		test_loader, SLP_rd_test, model, criterion,
		n_iter=n_iter, logger=logger, opts=opts, if_svVis=True)  # save preds, gt, preds_in ori, idst_normed to recovery
	pck_all = rst_test['pck']

	# perf_indicator = pck_all[-1][-1]  # last entry of list
	pckh05 = np.array(pck_all)[:, -1]        # why only 11 pck??
	titles_c = list(SLP_rd_test.joints_name[:SLP_rd_test.joint_num_ori]) + ['total']
	ut.prt_rst([pckh05], titles_c, ['pckh0.5'], fn_prt=logger.info)
	pth_rst = path.join(opts.rst_dir, opts.nmTest + '.json')
	with open(pth_rst, 'w') as f:
		json.dump(rst_test, f)

if __name__ == '__main__':
	main()