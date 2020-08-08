'''
combine images script.
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

opts = opt.parseArgs()
opts = opt.aug_opts(opts)

out_dir = r'S:\ACLab\rst_model\SLP\output_vis\single'  # for local
combFd = 'combined'
combFd = path.join(out_dir, combFd)
if not path.exists(combFd):
	os.makedirs(combFd)

li_sub = ['uncover', 'cover1', 'cover2']
li_file = os.listdir(path.join(out_dir, li_sub[0]))
N = len(li_file)
# N=1     # for test
for i in tqdm(range(N)):
	nm = li_file[i]
	# li_img = []
	# get idx_subj, idx_frm,
	idxs = [int(s) for s in re.findall(r'\d+', nm)] # not find
	# get RGB, uncover as first
	idx_subj = idxs[0]
	idx_frm = idxs[1]
	# print('processing subj {} frm {}'.format(idx_subj, idx_frm))
	RGB = getImg_dsPM(dsFd=opts.SLP_fd, idx_subj=idx_subj, modality='RGB', idx_frm=idx_frm)
	RGB = RGB[:,:,::-1]
	li_img = [RGB]
	for subFd in li_sub:
		li_img.append(cv2.imread(path.join(out_dir,subFd, nm)))
	img_hc = vis.hconcat_resize(li_img)
	pth_sv = path.join(combFd, nm)
	cv2.imwrite(pth_sv, img_hc)