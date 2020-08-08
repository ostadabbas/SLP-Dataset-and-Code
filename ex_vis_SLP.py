'''
script version for  SLP vis
'''
#%%
'''
For SLP visualization.
1. using sio to show depth map scatter plot for heatmp
'''
from data.SLP_RD import SLP_RD
import opt
# import skimage.io as skio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import trimesh as tm      # trimesh failed to show point cloud
# import open3d as o3d
import os
import os.path as path
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as sig
from utils import vis
import json
from PIL import Image


#%%
opts = opt.parseArgs()
id_machine = 0  # 0 for local, 1 for AR,  2 for cluster
# out dir for multiview only
if 0 == id_machine:
	out_dir = r'S:\ACLab\rst_model\SLP\output_vis\multi'  # for local
	opts.ds_fd = r'S:\ACLab\datasets'
elif 1 == id_machine:
	opts.ds_fd = '/home/jun/datasets'
	out_dir = '/home/jun/exp/SLP/output_vis/multi'  # for multi view
	pass
else:
	opts.ds_fd = '/scratch/liu.shu/datasets'
	out_dir = 'output_vis'

opts= opt.aug_opts(opts)
SLP_rd = SLP_RD(opts, phase='test')

idx = 0
arr, jt, bb = SLP_rd.get_array_joints(idx_smpl=idx)
ptc = SLP_rd.get_ptc(idx=idx)   # all pixel
arr_IR2depth = SLP_rd.get_array_A2B(idx=idx, modA='IR', modB='depthRaw')
arr_PM2depth = SLP_rd.get_array_A2B(idx=idx, modA='PM', modB='depthRaw')
sz_d = SLP_rd.sz_depth
f_d = SLP_rd.f_d
c_d = SLP_rd.c_d

#%%
# plotting, very slow
# fig = plt.figure(figsize=(15,10))
# ax = plt.axes(projection='3d')
#
# STEP = 1
# for x in range(0, arr.shape[0], STEP):
#     for y in range(0, arr.shape[1], STEP):
#         ax.scatter(
#             arr[x,y], y, x)
#              # c=[tuple(img[x, y, :3]/255)], s=3)    # for color
#     ax.view_init(15, 165)

#%%
# test trimesh      -- can't show point cloud
# ptc_tm = tm.points.PointCloud(ptc)
# ptc_tm.show(resolution=(800, 600), distance=2000)

#%% open3d vis, no transparency
# # pseudo color
# arr_IR = cv2.applyColorMap(arr_IR2depth, cv2.COLORMAP_HOT)
# arr_IR = arr_IR[:,:,::-1].copy()   # BGR -> RGB? ,  must rebuilt for raw buffer
# # alpha channel
# alpha = 50
# b_channel, g_channel, r_channel = cv2.split(arr_IR)
# alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * alpha  # creating a dummy alpha channel image.
# arr_IR = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
# arr_PM = cv2.applyColorMap(arr_PM2depth, cv2.COLORMAP_JET)
# arr_PM = arr_PM[:,:,::-1].copy()   # BGR -> RGB? ,  must rebuilt for raw buffer
# # cv2.imshow('ir2d', arr_PM)    # image is correct  check wha tis wrong, data ?
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # dist = 3  # fake distance for PM map
# d_PM = 2700
# arr_depth_flat = np.ones_like(arr)*d_PM      # exact same thing?  ones like copy not working
# # arr_depth_flat[200:400, 200:400] = d_PM # can't be seen pure white
# IR_o3d = o3d.geometry.Image(arr_IR)
# PM_o3d = o3d.geometry.Image(arr_PM)
# depth_o3d = o3d.geometry.Image(arr)
# depth_PM_o3d = o3d.geometry.Image(arr_depth_flat)
# rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
# 	IR_o3d, depth_o3d, convert_rgb_to_intensity=False)
# cam_intr = o3d.camera.PinholeCameraIntrinsic(width=sz_d[0], height=sz_d[1], fx=f_d[0], fy=f_d[1], cx=c_d[0], cy=c_d[1])
# # try depth only
# # pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, cam_intr)
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam_intr)
# pcd_pm = o3d.geometry.PointCloud.create_from_rgbd_image(pmd_img, cam_intr)
# # Flip it, otherwise the pointcloud will be upside down
# extrinsic = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
# pcd.transform(extrinsic)
# pcd_pm.transform(extrinsic)
# o3d.visualization.draw_geometries([pcd])  # can't see PM data??


#%% pyplot version
# vis.vis_IR_D_PM(arr, arr_IR2depth, arr_PM2depth)
pth = 'tmp/test_sv.png'
# img_cv2 = vis.vis_IR_D_PM(arr, arr_IR2depth, arr_PM2depth, pth=None)
vis.vis_IR_D_PM(arr, arr_IR2depth, arr_PM2depth, pth=pth)   # test save is working
# from IPython.display import Image
# img_ipy= Image(img_bytes)

# img_PIL = Image.frombytes(img_bytes)  # need: mode, size, data
# img_PIL.show()
# arr = np.array(img_ipy)
# load
# json can't work
# if pth.endswith('.json'):
# 	with open(pth, 'r') as f:
# 		dtIn = json.load(f)
# 		img = np.array(json.load(f))
# else:
# 	img = cv2.imread(pth)
# cv2.imshow('load img', img_cv2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# raw script for IR-D-PM vis
# PM_max = 100
# d_PM = 500
# opacity_IR = 0.5
# opacity_PM = 0.8
# d_bed = 2100    # distance to bed
# arr_plt = arr[80:420, 160:340]  # cropped      w 180 h 340  mid 90   -80 - 260
# IR_plt = arr_IR2depth[80:420, 160:340]
# PM_plt = arr_PM2depth[80:420, 160:340]
# # PM_plt = cv2.normalize(PM_plt, None, alpha=0, beta=PM_max, norm_type=cv2.NORM_MINMAX)
# PM_plt = np.clip(PM_plt, 0, PM_max) # clip to narrow range
# shp = arr_plt.shape # 512 x 424
# x, y = np.mgrid[0:shp[0], 0:shp[1]]
# # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=arr)]) # can't work empty
# fig = go.Figure(data=[go.Scatter3d(x=x.flatten(), y=y.flatten(), z=-arr_plt.flatten().astype(int), mode='markers', marker=dict(
# 	size=1,
# 	color=IR_plt.flatten(),  # set color to an array/list of desired values
# 	colorscale='Hot',  # choose a colorscale
# 	opacity=opacity_IR
# ))]) # can't work empty
# # add PM surface
# Z_pm = np.ones(shp)*(-d_bed-d_PM)     # 2264 bed height , higher to cut
# fig.add_surface(x=x, y=y, z=Z_pm,
#                 surfacecolor=PM_plt,
#                 colorscale='Jet',
#                 showscale=False,
#                 connectgaps=True,
#                 opacity=opacity_PM)
# # dict: nticks
# fig.update_layout(
# 	scene=dict(
# 		xaxis=dict(range=[0, 340], showticklabels=False),
# 		yaxis=dict(range=[-80, 260], showticklabels=False),
# 		zaxis=dict(range=[-2700, -1000], showticklabels=False)),
# 	width=700,
# 	margin=dict(r=20, l=10, b=10, t=10))          # x, y, z are r, c, z direction
#
# # surface version,  narrow down could help but pt cloud seems better
# # arr_plt = arr[80:420, 160:340] # cropped
# # arr_plt = sig.medfilt(arr_plt, kernel_size=9)
# # # fig = go.Figure(data=[go.Surface(z=-arr_plt.astype(int))])
# # fig = go.Figure(data=[go.Surface(z=-arr.astype(int))])
# # fig.show()
# # test the save image version
# sv_fd = 'tmp'
# if not os.path.exists(sv_fd):
# 	os.makedev(sv_fd)
# # fig = go.FigureWidget(data=go.Bar(y=[2, 3, 1]))
# fig.write_image(path.join(sv_fd, 'fig_test.png'))