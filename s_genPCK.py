'''
generate the pck plot from the candidate results json
'''
import numpy as np
import matplotlib.pyplot as plt
import json
from os import path
from utils import vis

def Cvt2eight(fiv_matrix):
	'''
	This is hardwired to trnasfer the SLP skeleton to define the paired pck.
	Parameters
	----------
	fiv_matrix :  The matrix of original (14,11)

	Returns
	-------
	new_matrix : The matrix after change (8,11)

	'''
	new_matrix = np.zeros((8, 11))
	new_matrix[0, :] = (fiv_matrix[0, :] + fiv_matrix[5, :]) / 2
	new_matrix[1, :] = (fiv_matrix[1, :] + fiv_matrix[4, :]) / 2
	new_matrix[2, :] = (fiv_matrix[2, :] + fiv_matrix[3, :]) / 2
	new_matrix[3, :] = (fiv_matrix[6, :] + fiv_matrix[11, :]) / 2
	new_matrix[4, :] = (fiv_matrix[7, :] + fiv_matrix[10, :]) / 2
	new_matrix[5, :] = (fiv_matrix[8, :] + fiv_matrix[9, :]) / 2
	new_matrix[6, :] = fiv_matrix[13, :]
	new_matrix[7, :] = fiv_matrix[14, :]
	return new_matrix

# settings
outFd = '/home/liu.shu/codesPool/SLP/output'        # output folder
dct_nm2conf={
	'HRpose': 'Sun,CVPR\'19',
	'RESpose':'Xiao,ECCV\'18',
	'ChainedPredictions': 'Gkioxari, ECCV\'16',
	'PoseAttention': 'Chu,CVPR\'17',
	'PyraNet': 'Yang,ICCV\'17',
	'StackedHourGlass': 'Newell,ECCV\'16',
}
ticks = np.linspace(0, 0.5, 11)  # 11 steps
if_jts = True  # if plot jts specific plots.

## cross mthds comparison
#---  settings
if not if_jts:
	# SLP_set = 'simLab'
	# layout = [1, 3]
	# mods = ['IR', 'depth', 'PM']  # could be depth or  depth-IR-PM
	# give the names of the plotting method
	SLP_set = 'simLab'
	layout = [1, 2]
	mods = ['IR', 'depth']  # could be depth or  depth-IR-PM
	nms_mth = [
		'Sun, CVPR\'19',
		'Xiao, ECCV\'18',
		'Gkioxari, ECCV\'16',
		'Chu, CVPR\'17',
		'Yang, ICCV\'17',
		'Newell, ECCV\'16'
	]
	# give the names for candidate  methods
	nms_rst = [
		'SLP_{}_u12_HRpose_exp',
		'SLP_{}_u12_RESpose_exp',
		'SLP_{}_u12_ChainedPredictions_exp',
		'SLP_{}_u12_PoseAttention_exp',
		'SLP_{}_u12_PyraNet_exp',
		'SLP_{}_u12_StackedHourGlass_exp',
	]
	svNm = 'total_{}.pdf'.format(SLP_set)  # depth_total_danaLab.pdf
	nm_test = 'SLP-{}_exp.json'.format(SLP_set)  # indicate the test name
	# nms_rst = [nm.format(mod) for nm in nms_rst]        # add in mod
	nms_jt = mods[:]    # copy it
	nms_jt[0] = 'LWIR'      # specially treated
	li_rsts = []  # keep the result     n_curve:  n_jt(plots)  x  14
	for nm_rst in nms_rst:
		li_rst_mth = []     # for current method
		for mod in mods:
			nm_rst_t= nm_rst.format(mod)
			pth = path.join(outFd, nm_rst_t, 'result', nm_test)
			with open(pth) as f:
				dict_t = json.load(f)
			mat_cb = Cvt2eight(np.array(dict_t['pck']))  # 14 -> 8 x 11
			# print('total pck for is', mat_cb[-1])
			li_rst_mth.append(mat_cb[-1])  # extend dimension

		li_rsts.append(np.array(li_rst_mth))        # each mat has 3
else:
	## cross mthds comparison
	# ---  settings
	# SLP_set = 'simLab'
	# layout = [1, 3]
	# mods = ['IR', 'depth', 'PM']  # could be depth or  depth-IR-PM
	# give the names of the plotting method
	nm_mdl = 'HRpose'       # plot individual model
	SLP_set = 'danaLab'
	layout = [2, 4]
	figsize= [20, 8]
	mods = ['IR', 'depth', 'PM', 'PM-depth', 'PM-IR', 'IR-depth', 'PM-depth-IR']  # could be depth or  depth-IR-PM
	nms_mth = [
		'LWIR', 'depth', 'PM', 'PM-depth', 'PM-LWIR', 'LWIR-depth', 'PM-depth-LWIR'
	]
	# give the names for candidate  methods
	#--- for HRpose only
	# nms_rst = [
	# 	'SLP_{}_u12_HRpose_exp',
	# 	'SLP_{}_u12_RESpose_exp',
	# 	'SLP_{}_u12_ChainedPredictions_exp',
	# 	'SLP_{}_u12_PoseAttention_exp',
	# 	'SLP_{}_u12_PyraNet_exp',
	# 	'SLP_{}_u12_StackedHourGlass_exp',
	# ]

	nm_rst = 'SLP_{}_u12_' + nm_mdl + '_exp'
	svNm = 'jts_{}_{}.pdf'.format(nm_mdl, SLP_set)  # depth_total_danaLab.pdf
	nm_test = 'SLP-{}_exp.json'.format(SLP_set)  # indicate the test name
	# nms_rst = [nm.format(mod) for nm in nms_rst]        # add in mod
	li_rsts = []  # keep the result     n_curve:  n_jt(plots)  x  14
	# print('ticks is', ticks)
	for mod in mods:
		nm_rst_t = nm_rst.format(mod)
		pth = path.join(outFd, nm_rst_t, 'result', nm_test)
		with open(pth) as f:
			dict_t = json.load(f)
		mat_cb = Cvt2eight(np.array(dict_t['pck']))  # 14 -> 8 x 11
		li_rsts.append(mat_cb)
	nms_jt = [
		'Ankle', 'Knee', 'Hip', 'Wrist', 'Elbow', 'Shoulder', 'Head',
		'Total'
	]


# nm_rst = 'SLP-simLab_exp.json'    # for dana Lab test

# for loop gen pth names,
# transfer to 8, make a list of result
# call genPCK(li_mat, ticks,  nms_mth,  nms_jt,  outFd)  # return image or save


vis.genPCK(li_rsts, ticks, nms_mth, nms_jt, svNm=svNm, layout=layout, figsize=figsize)




