'''
model script for test
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
import tqdm
# large opts

def main():
	opts = opt.parseArgs()
	opt.print_options(opts)
	model_dir = 'output/A2J/model_dump/'    # hardwired due in this test case
	# feeder
	SLP_rd = SLP_RD(opts, phase='test')       #
	SLP_fd = SLP_FD(SLP_rd, opts)



	# visualize images
	# idx = 0
	# pch, jt = SLP_fd.SLP_A2J(idx)
	# pch = pch.transpose([1,2,0])    # to cv2 format
	# pch = np.concatenate((pch,)*3, axis=-1)
	# # pch_2d = pch[:,:,0]
	# pch_nm = ut.normImg(pch.astype(float))
	# jt[:,2] = 1 - jt[:, 2]
	# im_show = vis.vis_keypoints(pch_nm, jt.T,  SLP_rd.skels_idx)
	# cv2.imshow('image check', im_show)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# model

	test_dataloaders = torch.utils.data.DataLoader(SLP_fd, batch_size=opts.batch_size,
	                                               shuffle=False, num_workers=8)
	dl_iter = iter(test_dataloaders)
	itr_per_epoch = math.ceil(
		SLP_fd.__len__() / opts.num_gpus / opts.batch_size)  # single ds test on batch share
	if opts.testIter > 0:
		itr_per_epoch = min(itr_per_epoch, opts.testIter)

	net = model.A2J_model(num_classes=14)
	net.load_state_dict(torch.load(model_dir))
	net = net.cuda()
	net.eval()
	# save the jts, pth_desc
	output = torch.FloatTensor()
	with torch.no_grad():
		for i in tqdm(range(itr_per_epoch)):        # loop all test
			img, label = next(dl_iter)
			heads = net(img)
			pred_keypoints = post_precess(heads, voting=False)  #
			output = torch.cat([output, pred_keypoints.data.cpu()], 0)  # N x n_jt x  3 (x,y:pix, depth:mm)

	result = output.cpu().data.numpy()
	## idx test
	idx_show = 0
	jt = result[0]



if __name__ == '__main__':
	main()