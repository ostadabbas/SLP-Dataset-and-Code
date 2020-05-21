'''
for key points visualization. Also visualizer for visdom class.
'''
import os
import os.path as osp
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import ntpath
import time
# from . import utils_tool, html
from subprocess import Popen, PIPE
# from scipy.misc import imresize
from skimage.transform import resize  # misc deprecated e
from skimage import io, transform, img_as_ubyte
from .utils import make_folder
from . import utils as utils_tool



def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
	'''
	column format
	:param img:
	:param kps: 3 * n_jts changed to n_jts x 3
	:param kps_lines:
	:param kp_thresh:
	:param alpha:
	:return:
	'''
	# Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
	kps = kps.T # transfrom it
	cmap = plt.get_cmap('rainbow')
	colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
	colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

	# Perform the drawing on a copy of the image, to allow for blending.
	kp_mask = np.copy(img)

	# Draw the keypoints.
	for l in range(len(kps_lines)):
		i1 = kps_lines[l][0]
		i2 = kps_lines[l][1]
		p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
		p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
		if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
			cv2.line(
				kp_mask, p1, p2,
				color=colors[l], thickness=2, lineType=cv2.LINE_AA)
		if kps[2, i1] > kp_thresh:
			cv2.circle(
				kp_mask, p1,
				radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
		if kps[2, i2] > kp_thresh:
			cv2.circle(
				kp_mask, p2,
				radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

	# Blend the keypoints.
	return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_3d(kpt_3d, skel, kpt_3d_vis=None, sv_pth=None, rg=None, fig_id = 1):
	'''
	simplified version with less positional input comparing to vis pack.  Just show the skeleton, if non visibility infor, show full skeleton. Plot in plt, and save it.
	:param kpt_3d:  n_jt * 3
	:param skel:
	:param kpt_3d_vis:
	:param sv_pth: if not given then show the 3d figure.
	:param rg: the range for x, y and z in  ( (xs, xe), (ys, ye), (zs, ze)) format
	:return:
	'''

	fig = plt.figure(fig_id)
	ax = fig.add_subplot(111, projection='3d')
	# Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
	cmap = plt.get_cmap('rainbow')
	colors = [cmap(i) for i in np.linspace(0, 1, len(skel) + 2)]
	colors = [np.array((c[2], c[1], c[0])) for c in colors]

	if not kpt_3d_vis:
		kpt_3d_vis = np.ones((len(kpt_3d), 1))  # all visible

	for l in range(len(skel)):
		i1 = skel[l][0]
		i2 = skel[l][1]
		x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
		y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
		z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

		if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
			ax.plot(x, z, -y, c=colors[l], linewidth=2)
		if kpt_3d_vis[i1, 0] > 0:
			ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=[colors[l]], marker='o')
		if kpt_3d_vis[i2, 0] > 0:
			ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=[colors[l]], marker='o')

	ax.set_title('3D vis')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Z Label')
	ax.set_zlabel('Y Label')

	if rg:
		ax.set_xlim(rg[0])      # x
		ax.set_zlim([-e for e in rg[1]][::-1])   # - y
		ax.set_ylim(rg[2])

	# ax.set_xlim([0,cfg.input_shape[1]])
	# ax.set_ylim([0,1])
	# ax.set_zlim([-cfg.input_shape[0],0])
	# ax.legend()       # no legend
	if not sv_pth:  # no path given, show image, otherwise save
		plt.show()
	else:
		fig.savefig(sv_pth, bbox_inches='tight')
	plt.close(fig)  # clean after use


def vis_entry(entry_dict):
	'''
	from the entry dict plot the images
	:param entry_dict:
	:return:
	'''


if sys.version_info[0] == 2:
	VisdomExceptionBase = Exception
else:
	VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
	"""Save images to the disk. Also to webpage

	Parameters:
		webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
		visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
		image_path (str)         -- the string is used to create image paths
		aspect_ratio (float)     -- the aspect ratio of saved images
		width (int)              -- the images will be resized to width x width

	This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
	"""
	image_dir = webpage.get_image_dir()
	short_path = ntpath.basename(image_path[0])
	name = os.path.splitext(short_path)[0]

	webpage.add_header(name)
	ims, txts, links = [], [], []

	for label, im_data in visuals.items():
		im = utils_tool.tensor2im(im_data)
		image_name = '%s_%s.png' % (name, label)
		save_path = os.path.join(image_dir, image_name)
		h, w, _ = im.shape
		if aspect_ratio > 1.0:
			im = resize(im, (h, int(w * aspect_ratio)))
		if aspect_ratio < 1.0:
			im = resize(im, (int(h / aspect_ratio), w))
		utils_tool.save_image(im, save_path)

		ims.append(image_name)
		txts.append(label)
		links.append(image_name)
	webpage.add_images(ims, txts, links, width=width)


def ipyth_imshow(img):
	# use ipython to show an image
	import cv2
	import IPython
	_, ret = cv2.imencode('.jpg', img)
	i = IPython.display.Image(data=ret)
	IPython.display.display(i)

def vis_3d(kpt_3d, skel, kpt_3d_vis=None, sv_pth=None, rg=None, fig_id = 1):
	'''
	simplified version with less positional input comparing to vis pack.  Just show the skeleton, if non visibility infor, show full skeleton. Plot in plt, and save it.
	:param kpt_3d:  n_jt * 3
	:param skel:
	:param kpt_3d_vis:
	:param sv_pth: if not given then show the 3d figure.
	:param rg: the range for x, y and z in  ( (xs, xe), (ys, ye), (zs, ze)) format
	:return:
	'''

	fig = plt.figure(fig_id)
	ax = fig.add_subplot(111, projection='3d')
	# Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
	cmap = plt.get_cmap('rainbow')
	colors = [cmap(i) for i in np.linspace(0, 1, len(skel) + 2)]
	colors = [np.array((c[2], c[1], c[0])) for c in colors]

	if not kpt_3d_vis:
		kpt_3d_vis = np.ones((len(kpt_3d), 1))  # all visible

	for l in range(len(skel)):
		i1 = skel[l][0]
		i2 = skel[l][1]
		x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
		y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
		z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

		if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
			ax.plot(x, z, -y, c=colors[l], linewidth=2)
		if kpt_3d_vis[i1, 0] > 0:
			ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=[colors[l]], marker='o')
		if kpt_3d_vis[i2, 0] > 0:
			ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=[colors[l]], marker='o')

	ax.set_title('3D vis')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Z Label')
	ax.set_zlabel('Y Label')

	if rg:
		ax.set_xlim(rg[0])      # x
		ax.set_zlim([-e for e in rg[1]][::-1])   # - y
		ax.set_ylim(rg[2])

	# ax.set_xlim([0,cfg.input_shape[1]])
	# ax.set_ylim([0,1])
	# ax.set_zlim([-cfg.input_shape[0],0])
	# ax.legend()       # no legend
	if not sv_pth:  # no path given, show image, otherwise save
		plt.show()
	else:
		fig.savefig(sv_pth, bbox_inches='tight')
	plt.close(fig)  # clean after use

def vis_3d_cp(kpt_3d_li, skel, kpt_3d_vis=None, sv_pth=None, rg=None, fig_id = 1):
	'''
	visulize the 3d plot in one figure for compare purpose, with differed color
	:param kpt_3d:  n_jt * 3
	:param skel:
	:param kpt_3d_vis:
	:param sv_pth: if not given then show the 3d figure.
	:param rg: the range for x, y and z in  ( (xs, xe), (ys, ye), (zs, ze)) format
	:return:
	'''
	if isinstance(kpt_3d_li, np.ndarray):
		kpt_3d_li =[ kpt_3d_li] # to list
	N = len(kpt_3d_li)
	fig = plt.figure(fig_id)
	ax = fig.add_subplot(111, projection='3d')
	# Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
	cmap = plt.get_cmap('rainbow')
	colors = [cmap(i) for i in np.linspace(0, 1, N)]
	colors = [np.array((c[2], c[1], c[0])) for c in colors]
	if not kpt_3d_vis:
		kpt_3d_vis = np.ones((len(kpt_3d_li[0]), 1))  # all visible

	for i, kpt_3d in enumerate(kpt_3d_li):
		for l in range(len(skel)):
			i1 = skel[l][0]
			i2 = skel[l][1]
			x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
			y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
			z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

			if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
				ax.plot(x, z, -y, c=colors[i], linewidth=2)
			if kpt_3d_vis[i1, 0] > 0:
				ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=[colors[i]], marker='o')
			if kpt_3d_vis[i2, 0] > 0:
				ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=[colors[i]], marker='o')

	ax.set_title('3D vis')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Z Label')
	ax.set_zlabel('Y Label')

	if rg:
		ax.set_xlim(rg[0])      # x
		ax.set_zlim([-e for e in rg[1]][::-1])   # - y
		ax.set_ylim(rg[2])

	# ax.set_xlim([0,cfg.input_shape[1]])
	# ax.set_ylim([0,1])
	# ax.set_zlim([-cfg.input_shape[0],0])
	# ax.legend()       # no legend
	if not sv_pth:  # no path given, show image, otherwise save
		plt.show()
	else:
		fig.savefig(sv_pth, bbox_inches='tight')
	plt.close(fig)  # clean after use


def showJoints(img, joint_img, svPth = None):
	'''
	label all joints to help figure out joint name
	:param img:
	:param joint_img: n_jt *3 or n_jt *2
	:return:
	'''
	joint_img = joint_img.astype(int)
	h, w = img.shape[:2]
	offset = 0
	cycle_size = min(1, h/100)
	for i, joint in enumerate(joint_img):
		cv2.circle(img, (joint[0], joint[1]), cycle_size, (0, 255, 0), -1)
		cv2.putText(img, str(i), (joint[0] + offset, joint[1] + offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
	if not svPth:
		cv2.imshow('label joints', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		cv2.imwrite(svPth, img)

def save_2d_tg3d(img_patch, pred_2d, skel, sv_dir, idx='tmp', suffix=''):
	'''
	make joint labeled folder in image, save image into sv_dir/2d/idx.jpg
	:param img_patch: image suppose to be c,w,h rgb numpy
	:param pred_2d: x,y, score  3xn_jt
	:param sv_dir:  where to save
	:return:
	'''
	sv_dir = osp.join(sv_dir, '2d'+suffix)
	make_folder(sv_dir)
	tmpimg = vis_keypoints(img_patch, pred_2d, skel)
	cv2.imwrite(osp.join(sv_dir, str(idx) + '.jpg'), tmpimg)


def save_3d_tg3d(kpt_3d, sv_dir, skel, idx='tmp', suffix=None):
	'''
	save 3d plot to designated places. tg3d task generalization for 3d
	:param coord_out:
	:param sv_dir:
	:param skel:
	:param idx:
	:param suffix:
	:return:
	'''
	rg = None
	if suffix:
		svNm = '3d_' + suffix
		if 'hm' == suffix:
			rg = ((0,64),) * 3
		elif 'A2J' == suffix:
			rg = [[-1, 1], [-1, 1], [2, 4]]
		else:
			rg = ((-1000, 1000), ) * 3
	else:
		svNm = '3d'
	sv_dir = osp.join(sv_dir, svNm)
	make_folder(sv_dir)
	sv_pth = osp.join(sv_dir, str(idx) + '.jpg')
	vis_3d(kpt_3d, skel, sv_pth=sv_pth, rg=rg)


def save_hm_tg3d(HM, sv_dir, n_jt=17, idx='tmp', if_cmap=True):
	'''
	transfer 3d heatmap into front view and side view
	:param HM:  cxhxw  format numpy possibly  0~1  (64x17) * 64 * 64
	:param sv_dir:
	:param idx:
	:return:
	'''
	sv_dir = osp.join(sv_dir, 'hm')
	make_folder(sv_dir)

	# to each jt  # reshape change itself?
	depth_dim = int(HM.shape[0]/n_jt)
	hm = HM.copy().reshape([n_jt, depth_dim, *HM.shape[1:]])
	hm_xy_li = []
	hm_yz_li = []
	for i in range(n_jt):
		hm_xy = hm[i].mean(axis=0)  # channel first
		hm_yz = hm[i].mean(axis=2)  # along the x direction or r direction
		hm_xy_li.append(hm_xy)
		hm_yz_li.append(hm_yz)
		if if_cmap:
			cmap = plt.cm.jet
			norm = plt.Normalize(vmin=hm_xy.min(), vmax=hm_xy.max())
			hm_xy = cmap(norm(hm_xy))
			norm = plt.Normalize(vmin=hm_yz.min(), vmax=hm_yz.max())
			hm_yz = cmap(norm(hm_yz))
		io.imsave(osp.join(sv_dir, 'f{}_jt{}.png'.format(idx, i)), img_as_ubyte(hm_xy))
		io.imsave(osp.join(sv_dir, 's{}_jt{}.png'.format(idx, i)), img_as_ubyte(hm_yz))
	# for total
	hm_xy_tot = np.mean(hm_xy_li, axis=0)
	hm_yz_tot = np.mean(hm_yz_li, axis=0)
	if if_cmap:
		cmap = plt.cm.jet
		norm = plt.Normalize(vmin=hm_xy_tot.min(), vmax=hm_xy_tot.max())
		hm_xy_tot = cmap(norm(hm_xy_tot))
		norm = plt.Normalize(vmin=hm_yz_tot.min(), vmax=hm_yz_tot.max())
		hm_yz_tot = cmap(norm(hm_yz_tot))
	io.imsave(osp.join(sv_dir, 'f{}_tot.png'.format(idx, i)), img_as_ubyte(hm_xy_tot))
	io.imsave(osp.join(sv_dir, 's{}_tot.png'.format(idx, i)), img_as_ubyte(hm_yz_tot))


def save_Gfts_raw_tg3d(G_fts, sv_dir, idx='tmp'):
	'''
	save all G_fts in a raw npy format to for recovery later.
	:param G_fts: already is numpy.
	:param sv_dir_G:
	:param idx:
	:param shape: what grid is needed,  first prod(shape) elements will be used to form grid
	:param out_sz: the output size of the feature map to make it large
	:return:
	'''
	sv_dir_G = osp.join(sv_dir, 'G_fts_raw')
	make_folder(sv_dir_G)
	if type(G_fts) is list:
		for i, G_ft in enumerate(G_fts):
			np.save(osp.join(sv_dir_G, str(idx) + '_' + str(i) + '.npy'), G_fts)    # idx_iLayer.jpg formate
	else:
		np.save(osp.join(sv_dir_G, str(idx)+'.npy'), G_fts)

def save_Gfts_tg3d(G_fts, sv_dir, idx='tmp', shape=(5, 5), out_sz=(64, 64)):
	'''

	:param G_fts:
	:param sv_dir_G:
	:param idx:
	:param shape: what grid is needed,  first prod(shape) elements will be used to form grid
	:param out_sz: the output size of the feature map to make it large
	:return:
	'''
	sv_dir_G = osp.join(sv_dir, 'G_fts')
	make_folder(sv_dir_G)
	n = np.prod(shape)
	if type(G_fts) is list:     # for list case
		for i, G_ft in enumerate(G_fts):
			fts = G_ft[:n]  # only first few
			n_cols = shape[1]
			# resize the fts (c last , resize, c back)
			fts_rsz = transform.resize(fts.transpose((1, 2, 0)), out_sz).transpose((2, 0, 1))
			# gallery
			grid = gallery(fts_rsz, n_cols=n_cols)
			#  cmap = plt.cm.jet        # can also color map it
			# save
			norm = plt.Normalize(vmin=grid.min(), vmax=grid.max())
			io.imsave(osp.join(sv_dir_G, str(idx) + '_' + str(i) + '.png'), img_as_ubyte(norm(grid)))

			# for histogram
			sv_dir_hist = osp.join(sv_dir, 'hist')
			make_folder(sv_dir_hist)
			# hist_G = np.histogram(G_fts)
			plt.clf()
			fts_hist = G_fts.flatten()
			fts_hist = fts_hist[fts_hist > 0.1]
			plt.hist(fts_hist, bins=50)
			plt.savefig(osp.join(sv_dir_hist, str(idx) + '_' + str(i) + '.png'))

	else:
		fts = G_fts[:n]  # only first few
		n_cols = shape[1]
		# resize the fts (c last , resize, c back)
		fts_rsz = transform.resize(fts.transpose((1, 2, 0)), out_sz).transpose((2, 0, 1))
		# gallery
		grid = gallery(fts_rsz, n_cols=n_cols)
		#  cmap = plt.cm.jet        # can also color map it
		# save
		norm = plt.Normalize(vmin=grid.min(), vmax=grid.max())
		io.imsave(osp.join(sv_dir_G, str(idx) + '.png'), img_as_ubyte(norm(grid)))

		# for histogram
		sv_dir_hist = osp.join(sv_dir, 'hist')
		make_folder(sv_dir_hist)
		# hist_G = np.histogram(G_fts)
		plt.clf()
		fts_hist = G_fts.flatten()
		fts_hist = fts_hist[fts_hist>0.1]
		plt.hist(fts_hist, bins=50)
		plt.savefig(osp.join(sv_dir_hist, str(idx) + '.png'))

def gallery(array, n_cols=5):
	nindex, height, width = array.shape[:3]
	shp = array.shape
	if len(shp) > 3:
		if_clr = True
		intensity = shp[3]
	else:
		if_clr = False
	nrows = nindex // n_cols
	assert nindex == nrows * n_cols
	# want result.shape = (height*nrows, width*ncols, intensity)
	# shp_new = [nrows,ncols, height, width] + shp[3:]
	if if_clr:
		result = (array.reshape(nrows, n_cols, height, width, intensity)
		          .swapaxes(1, 2)
		          .reshape(height * nrows, width * n_cols, intensity))
	else:
		result = (array.reshape(nrows, n_cols, height, width)
		          .swapaxes(1, 2)
		          .reshape(height * nrows, width * n_cols))
	return result

