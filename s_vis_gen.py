'''
generate the visual images
'''
import opt
from data.SLP_RD import SLP_RD
import opt
from utils import vis
import tqdm
from os import path
import os
from tqdm import tqdm
from math import pi, cos, sin
# from threading import Thread
#
# class THRD_sv_IRD_PM(Thread):     # can't work to communicate with plotly servfer
# 	def __init__(self, name, D, IR, PM, pth, eye=[1.0, -1.25, 0.7]):
# 		Thread.__init__(self)
# 		self.name = name
# 		self.D = D
# 		self.IR = IR
# 		self.PM = PM
# 		self.pth = pth
# 		self.eye = eye
#
# 	def run(self):
# 		# Get lock to synchronize threads
# 		# threadLock.acquire()
# 		# print_time(self.name, self.counter, 3)
# 		# Free lock to release next thread
# 		# threadLock.release()
# 		vis.vis_IR_D_PM(self.D, self.IR, self.PM, pth=self.pth) # save out directly


rt_smpl = 15 # every 5 image draw a img
# out_dir = 'output_vis/single'
# out_dir = r'S:\ACLab\rst_model\SLP\output_vis\single'   # for local
out_dir = r'S:\ACLab\rst_model\SLP\output_vis\multi'   # for local
# multi setting
eye_dft = [1.0, -1.25, 0.7]
r_cam = 1.75
n_frm_rd = 60  # 10 frame for test
# fd settings

opts = opt.parseArgs()
id_machine = 0  # 0 for local, 1 for AR,  2 for cluster , can only local
# out dir for multiview only
if 0 == id_machine:
	out_dir = r'S:\ACLab\rst_model\SLP\output_vis\multi'  # for local
	opts.ds_fd = r'S:\ACLab\datasets'
elif 1 == id_machine:
	opts.ds_fd = '/home/jun/datasets'
	out_dir = '/home/jun/exp/SLP/output_vis/multi'     # for multi view
	pass
else:
	opts.ds_fd = '/scratch/liu.shu/datasets'
	out_dir = 'output'
opts.cov_li = ['uncover', 'cover1', 'cover2']
opts = opt.aug_opts(opts)


if out_dir.endswith('multi'):
	if_multi = True
else:
	if_multi = False
#

print('get cover_li', opts.cov_li)
SLP_rd = SLP_RD(opts, phase='test')

n_smpl = SLP_rd.n_smpl
# n_smpl = 270  #for testing      # for 2 people  135 * 2

print('saving image to {}'.format(out_dir))
li_thrd = []
if if_multi:
	print('run multi session')
else:
	print('run single session')
for idx in tqdm(range(270, n_smpl, rt_smpl), desc='Saving vis SLP'):
	arr_D, jt, bb = SLP_rd.get_array_joints(idx_smpl=idx)
	arr_IR2depth = SLP_rd.get_array_A2B(idx=idx, modA='IR', modB='depthRaw')
	arr_PM2depth = SLP_rd.get_array_A2B(idx=idx, modA='PM', modB='depthRaw')
	i_subj, cov, i_frm = SLP_rd.pthDesc_li[idx]
	bsNm = '{:05}_{:05}'.format(i_subj, i_frm)

	if if_multi:
		step = pi * 2 / n_frm_rd
		for i in range(n_frm_rd):
			# cam angle
			theta = step * i
			x = cos(theta) * r_cam
			y = sin(theta) * r_cam
			eye = [x, y, 0.7]
			# make folder  name/cov
			fd = path.join(out_dir, bsNm, cov)
			if not path.exists(fd):
				os.makedirs(fd)
			sv_pth = path.join(fd, 'image_{:05d}.png'.format(i))
			vis.vis_IR_D_PM(arr_D, arr_IR2depth, arr_PM2depth, pth=sv_pth, eye=eye) # single thread version

			# multi thread, orac server overwhelmed
			# thrdNm = '{}_{}_{}'.format(i_subj, cov, i_frm)
			# thrd = THRD_sv_IRD_PM(thrdNm, arr_D, arr_IR2depth, arr_PM2depth, sv_pth, eye)
			# thrd.start()
			# li_thrd.append(thrd)
	else:       # for single render
		fd=path.join(out_dir,  cov)
		if not path.exists(fd):
			os.makedirs(fd)
		sv_pth = path.join(fd, 'image_'+ bsNm+'.png')
		if False:
			print('save image to {}'.format(sv_pth))
		vis.vis_IR_D_PM(arr_D, arr_IR2depth, arr_PM2depth, pth=sv_pth)

# wait for all clear
# for t in li_thrd:
# 	t.join()
# print(' all thread completed')
