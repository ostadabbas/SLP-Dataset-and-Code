'''
combine images script. for multi folder
'''

import os
from os import path
import cv2
from utils import vis
from tqdm import tqdm
import plotly.io
import re
from data.SLP_RD import getImg_dsPM
import opt
from threading import Thread


opts = opt.parseArgs()
# set necessary here
opts.ds_fd = r'S:\ACLab\datasets'   # the local setting
opts = opt.aug_opts(opts)

out_dir = r'S:\ACLab\rst_model\SLP\output_vis\multi'  # for local
cb_fd_nm = 'combined'
vid_fd_nm = 'vid'
if_cb = True        # if combine images
if_vid = True        # if render video
fps = 5            # the fps for render

vidFd = path.join(out_dir, vid_fd_nm)
if not path.exists(vidFd):
	os.makedirs(vidFd)

li_sub = ['uncover', 'cover1', 'cover2']
li_frm = os.listdir(out_dir)   # subj_frm list
n_frm = len(li_frm)
# n_frm = 2       # for debug
for i0 in tqdm(range(n_frm)):
	frm_nm = li_frm[i0]
	frm_fd = path.join(out_dir, frm_nm)
	li_file = [file for file in os.listdir(path.join(frm_fd, li_sub[0])) if not file.endswith('.db')]   # get rid of the db file
	combFd = path.join(frm_fd, cb_fd_nm)

	print("save to combined folder {}".format(combFd))
	if not path.exists(combFd):
		os.makedirs(combFd)

	if if_cb:
		N = len(li_file)
		# N=1     # for test
		# get RGB for all multi views
		idxs = [int(s) for s in re.findall(r'\d+', frm_nm)]  # not find
		# get RGB, uncover as first
		idx_subj = idxs[0]
		idx_frm = idxs[1]
		# print('processing subj {} frm {}'.format(idx_subj, idx_frm))
		RGB = getImg_dsPM(dsFd=opts.SLP_fd, idx_subj=idx_subj, modality='RGB', idx_frm=idx_frm)
		RGB = RGB[:, :, ::-1]

		for i in range(N):
			nm = li_file[i]
			# li_img = []
			# get idx_subj, idx_frm,

			li_img = [RGB]
			for subFd in li_sub:
				pth = path.join(frm_fd, subFd, nm)
				# print('reading from pth {}'.format(pth))
				if path.exists(pth):
					li_img.append(cv2.imread(pth))
				else:
					print('{} is missing, can not continue'.format(pth))
					quit(-1)
			img_hc = vis.hconcat_resize(li_img)
			pth_sv = path.join(combFd, nm)
			print('save to {}'.format(pth_sv))
			cv2.imwrite(pth_sv, img_hc)

		if if_vid:
			vis.genVid(combFd, nm=frm_nm, svFd=vidFd,fps=fps, f_circle=2)
