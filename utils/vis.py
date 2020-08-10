'''
for key points visualization. Also visualizer for visdom class.
'''
import os
import os.path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import ntpath
from subprocess import Popen, PIPE
from skimage.transform import resize  # misc deprecated e
from skimage import io, transform, img_as_ubyte
from .utils import make_folder
from . import utils as utils_tool
import plotly.graph_objects as go


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
	kps = kps.T  # transfrom it
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


def vis_3d(kpt_3d, skel, kpt_3d_vis=None, sv_pth=None, rg=None, fig_id=1):
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
		ax.set_xlim(rg[0])  # x
		ax.set_zlim([-e for e in rg[1]][::-1])  # - y
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


def vis_3d_cp(kpt_3d_li, skel, kpt_3d_vis=None, sv_pth=None, rg=None, fig_id=1):
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
		kpt_3d_li = [kpt_3d_li]  # to list
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
		ax.set_xlim(rg[0])  # x
		ax.set_zlim([-e for e in rg[1]][::-1])  # - y
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


def showJoints(img, joint_img, svPth=None):
	'''
	label all joints to help figure out joint name
	:param img:
	:param joint_img: n_jt *3 or n_jt *2
	:return:
	'''
	joint_img = joint_img.astype(int)
	img_show = img.copy()
	h, w = img.shape[:2]
	offset = 0
	cycle_size = min(1, h / 100)
	for i, joint in enumerate(joint_img):
		cv2.circle(img_show, (joint[0], joint[1]), cycle_size, (0, 255, 0), -1)
		cv2.putText(img_show, str(i), (joint[0] + offset, joint[1] + offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
		            (255, 255, 255))
	if not svPth:
		return img_show
	else:
		cv2.imwrite(svPth, img)


def save_2d_skels(img_patch, pred_2d, skel, sv_dir, idx='tmp', suffix=''):
	'''
	make joint labeled folder in image, save image into sv_dir/2d/idx.jpg
	:param img_patch: image suppose to be c,w,h rgb numpy
	:param pred_2d: x,y, score  3xn_jt
	:param sv_dir:  where to save
	:return:
	'''
	sv_dir = osp.join(sv_dir, '2d' + suffix)
	make_folder(sv_dir)
	tmpimg = vis_keypoints(img_patch, pred_2d, skel)
	cv2.imwrite(osp.join(sv_dir, str(idx) + '.jpg'), tmpimg)


def save_img(img, sv_dir, idx='tmp', sub=''):
	'''save img to subFd'''
	sv_dir = osp.join(sv_dir, sub)
	make_folder(sv_dir)
	cv2.imwrite(osp.join(sv_dir, str(idx) + '.jpg'), img)


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
			rg = ((0, 64),) * 3
		elif 'A2J' == suffix:
			rg = [[-1, 1], [-1, 1], [2, 4]]
		else:
			rg = ((-1000, 1000),) * 3
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
	depth_dim = int(HM.shape[0] / n_jt)
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
			np.save(osp.join(sv_dir_G, str(idx) + '_' + str(i) + '.npy'), G_fts)  # idx_iLayer.jpg formate
	else:
		np.save(osp.join(sv_dir_G, str(idx) + '.npy'), G_fts)


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
	if type(G_fts) is list:  # for list case
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
		fts_hist = fts_hist[fts_hist > 0.1]
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


def vis_IR_D_PM(D, IR, PM, bb=[160, 80, 180, 340], PM_max=100, d_PM=500, opacity_IR=0.5, opacity_PM=0.8, d_bed=2100,
                eye=[1.0, -1.25, 0.7], pth=None):
	'''
	visualize the IR-D-PM in 3d point cloud format,
	:param D:   depth array  in unit16 (mm)
	:param IR: aligned IR image (unit8:255)
	:param PM: aligned PM image (uint8:255)
	:param bb: the cropping range of
	:param PM_max: max value of the PM to clip. For better visualization
	:param d_PM: the PM map away from the bed surface for better visual
	:param opacity_IR: the opacity IR_D
	:param opacity_PM: the opacity of PM
	:param d_bed: the distance of bed
	:param eye: the cam eye direction
	:param pth: the save path of the current image, if not given , show it directly
	:return:
	'''
	for i in range(len(bb)):  # cast to int
		bb[i] = int(bb[i])
	arr_plt = D[bb[1]:bb[1] + bb[3], bb[2]:bb[0] + bb[2]]  # cropped      w 180 h 340  mid 90   -80 - 260
	IR_plt = IR[bb[1]:bb[1] + bb[3], bb[2]:bb[0] + bb[2]]
	PM_plt = PM[bb[1]:bb[1] + bb[3], bb[2]:bb[0] + bb[2]]
	# PM_plt = cv2.normalize(PM_plt, None, alpha=0, beta=PM_max, norm_type=cv2.NORM_MINMAX)
	PM_plt = np.clip(PM_plt, 0, PM_max)  # clip to narrow range
	shp = arr_plt.shape  # 512 x 424
	x, y = np.mgrid[0:shp[0], 0:shp[1]]
	# fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=arr)]) # can't work empty
	fig = go.Figure(
		data=[go.Scatter3d(x=x.flatten(), y=y.flatten(), z=-arr_plt.flatten().astype(int), mode='markers', marker=dict(
			size=1,
			color=IR_plt.flatten(),  # set color to an array/list of desired values
			colorscale='Hot',  # choose a colorscale
			opacity=opacity_IR
		))])  # can't work empty
	# add PM surface
	Z_pm = np.ones(shp) * (-d_bed - d_PM)  # 2264 bed height , higher to cut
	fig.add_surface(x=x, y=y, z=Z_pm,
	                surfacecolor=PM_plt,
	                colorscale='Jet',
	                showscale=False,
	                connectgaps=True,
	                opacity=opacity_PM)
	# dict: nticks
	h = bb[3]
	w = bb[2]
	if h >= w:
		rg_x = [0, bb[3]]
		y_st = int((bb[2] - bb[3]) / 2)
		rg_y = [y_st, y_st + bb[3]]
	else:
		rg_y = [0, bb[1]]
		x_st = int((bb[3] - bb[2]) / 2)
		rg_x = [x_st, x_st + bb[1]]
	cam = dict(
		eye=dict(x=eye[0], y=eye[1], z=eye[2])
	)
	fig.update_layout(
		scene=dict(
			xaxis=dict(range=rg_x, title_text='', showticklabels=False, backgroundcolor="rgb(255, 255, 255)"),
			yaxis=dict(range=rg_y, title_text='', showticklabels=False, backgroundcolor="rgb(255, 255, 255)"),
			zaxis=dict(range=[-d_bed - 600, -d_bed + 1100], title_text='', showticklabels=False,
			           backgroundcolor="rgb(255, 255, 255)")),
		width=700,
		scene_camera=cam,
		margin=dict(r=0, l=0, b=0, t=0),  # no margin at all?
		# plot_bgcolor='white',        # not working
		# paper_bgcolor='white'
	)  # x, y, z are r, c, z direction

	if pth is None:
		fig.show()
	elif pth.endswith('.json'):
		fig.write_json(pth)
	else:
		fig.write_image(pth)  # orac version not working on cluster, use cv2


def hconcat_resize(im_list, if_maxh=True, interpolation=cv2.INTER_CUBIC):
	if if_maxh:  # use larger edge
		h_max = max(im.shape[0] for im in im_list)
		im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_max / im.shape[0]), h_max), interpolation=interpolation)
		                  for im in im_list]
	else:
		h_min = min(im.shape[0] for im in im_list)
		im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
		                  for im in im_list]
	return cv2.hconcat(im_list_resize)


def genVid(fd, nm=None, svFd='output/vid', fps=30, f_circle=1):
	'''
	from the target folder, generate the video with given fps to folder
	svFd with name of the fd last name.
	:param fd:
	:param svFd:
	:param fps:
	:param f_circle: circles to render. useful for cyclic rendered images.
	:return:
	'''
	if not os.path.exists(svFd):
		os.makedirs(svFd)
	if not nm:
		nm = os.path.basename(fd)  # reused for saming image
	f_li = os.listdir(fd)
	if not f_li:
		print('no images found in target dir')
		return
	img = cv2.imread(os.path.join(os.path.join(fd, f_li[0])))
	# make vid handle
	sz = (img.shape[1], img.shape[0])
	fourcc = cv2.VideoWriter_fourcc(*'avc1')
	video = cv2.VideoWriter(os.path.join(svFd, nm + '.mp4'), fourcc, fps, sz)
	N = len(f_li)
	N_rend = int(N * f_circle)  # the total number of the rendered image, circle the folder seq
	for i in range(N_rend):
		idx_img = i % N
		nm = f_li[idx_img]
		fname = os.path.join(os.path.join(fd, nm))
		img = cv2.imread(fname)
		video.write(img)
	video.release()


def genPCK(li_mat, ticks, nms_mth, nms_jt, outFd='output/pcks', svNm='pcks.pdf', layout=[2, 4], if_show=False, figsize=[20, 6], bt=0.2):
	'''
	generate pck from the list of mat result against the ticks. all pcks will be saved in one plot by with subplots format.
	:param li_mat: result list
	:param ticks:   the x axis
	:param nms_mth:     the names for legend, list of n
	:param nms_jt:      each line is a jt conrresponding result
	:param pckNm:       the save name of this plot
	:param layout:      sub plot layout
	:param outFd:       where to save pck
	:return:
	'''
	# create format
	font1 = {
		'family': 'Times New Roman',
		'size': 20}
	font2 = {
		'family': 'Times New Roman',
	         'size': 18}
	matShp = li_mat[0].shape
	# print('mat shape is', matShp)
	n_jt =len(nms_jt)
	assert layout[0]*layout[1] >= matShp[0], 'layout should have more plots than methods'
	assert matShp[1] == len(ticks), 'ticks {} should have same number as the input mat {}'.format_map(len(ticks), matShp[1])
	ncol = len(nms_mth)       # take ceil if odd
	if layout == [1, 1]:
		fig, axes = plt.subplots(figsiz=(12, 12))  # single one
		axes = [axes]       # to list
		ft_lgd = font2
	else:      # the size is specific for  2 x 4  version, can be adjusted dynamically in future
		fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)
		axes = axes.flatten()     #
		ft_lgd = font1

	for i in range(n_jt):      # loop all jt/subplots
		ax = axes[i]
		ax.set_title(nms_jt[i], fontdict=font2)
		for j, mat_t in enumerate(li_mat):        # j mth, i jt
			if i == n_jt - 1:  # if last
				ax.plot(ticks, mat_t[i], label = nms_mth[j], linewidth=3)
			else:
				ax.plot(ticks, mat_t[i], linewidth=3.0)
		ax.set_xlabel('Normalized distance', fontname='Times New Roman', fontsize=18)
		ax.set_ylabel('Detection Rate(%)', fontname='Times New Roman', fontsize=18)
		ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
		ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
		ax.tick_params(labelsize=14)

	pth_sv = os.path.join(outFd, svNm)
	lgd = fig.legend(loc="lower center",
	           # bbox_to_anchor=(0.5, -0.03),
	           ncol=ncol, prop=ft_lgd)
	plt.subplots_adjust(wspace=0.4, hspace=0.4,
	                    bottom=bt
	                    )
	if if_show:
		plt.show()
	fig.savefig(pth_sv, bbox_extra_artists=(lgd,), bbox_inches='tight')
	print('pck saved at {}'.format(pth_sv))
