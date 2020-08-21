'''
Parse arguments for training. act as config
'''
import os
import os.path as osp
import configargparse
import utils.utils as ut


def parseArgs(if_redef=True):
	parser = configargparse .ArgumentParser(formatter_class=configargparse .ArgumentDefaultsHelpFormatter)

	# -- env settings
	parser.add_argument('--ds_fd', default='/scratch/liu.shu/datasets',
		                    help='dataset directionry')  # for discovery
	parser.add_argument('--output_dir', default='output', help='default output dirs')  # code local.
	parser.add('--modelConf', default='config/HRpose.conf', is_config_file=True, help='Path to config file')
	parser.add_argument('--ifb_debug', action='store_true')
	parser.add_argument('--suffix_ptn_train', default='{model}')
	parser.add_argument('--suffix_exp_train', default='ts1', help='the manually given suffix for specific test')
	parser.add_argument('--suffix_ptn_test', default='{testset}-{SLP_set}',
	                    help='the suffix pattern to form name. Change accordingly to your exp (such as ds and methods ablation)')
	parser.add_argument('--suffix_exp_test', default='exp', help='mannualy added suffix for test result')

	# -- data setting
	parser.add_argument('--prep', default='SLP_A2J', help='data preparation method')
	parser.add_argument('--SLP_set', default='danaLab', help='[danaLab|simLab] for SLP section')
	parser.add_argument('--mod_src', nargs='+', default=['IR'],
	                    help='source modality list, can accept multiple modalities typical model [RGB|IR|depthRaw| PMarray]')
	parser.add_argument('--cov_li', nargs='+', default=['uncover', 'cover1', 'cover2'], help='the cover conditions')
	parser.add_argument('--fc_depth', type=float, default=50., help='the depth factor to pixel level')
	parser.add_argument('--if_bb', action='store_true', help='if use bounding box')
	# -- model
	parser.add_argument('--model', default='HRpose', help='model name [MPPE3D|A2J|HRpose|')   # the model to use
	parser.add_argument('--n_layers_D', type=int, default=3,
	                    help='descriminator layer number, for 8 bb, 2 layers are good')
	parser.add_argument('--net_BB', default='res50',
	                    help='backbone net type [res50|res101|res152], can extended to VGG different layers, not so important , add later')
	parser.add_argument('--out_shp', default=(64, 64, -1), type=int, nargs='+',
		                    help='the output(hm) size of the network, last dim is optional for 3d case')

	# -- train setting
	parser.add_argument('--trainset', nargs='+', default=['SLP'],
	                    help='give the main ds here the iter number will follow this one')  # to mod later
	# parser.add_argument('--if_D', default='y', help='if use discriminator, if single ds, then automatically set to n')
	parser.add_argument('--sz_pch', nargs='+', default=(256, 256), type=int, help='input image size, model 288, pix2pix 256')
	parser.add_argument('--end_epoch', default=100, type=int,
	                    help='when reach this epoch, will stop. python index style, your model will be saved as epoch_tar-1, ori 25 ')
	parser.add_argument('--epoch_step', default=-1, type=int,
	                    help='mainly for time constrained system, each time only train step epoches, -1 for all')
	parser.add_argument('--trainIter', default=-1, type=int,
	                    help='train iters each epoch, -1 for whole set. For debug purpose (DBP)')
	parser.add_argument('--optimizer', default='adam', help='[adam|nadam]')
	parser.add_argument('--lr', default=1e-3, type=float)
	parser.add_argument('--lr_policy', default='multi_step', help='[step|plateau|multi_step|cosine]')
	parser.add_argument('--lr_dec_epoch', nargs='+', type=int, default=[70, 90],
	                    help='the lr decay epoch, each time by decay factor ori 17,21')  # form string sec dec17-21 if needed
	parser.add_argument('--lr_dec_factor', default=0.1, type=float)
	parser.add_argument('--batch_size_pGPU', default=60, type=int, help='batch size per gpu')

	# test batch size 16 what is the reason,, no idea
	parser.add_argument('--gpu_ids', nargs='+', default=[0], type=int, help='the ids of the gpu')
	# parser.add_argument('--if_coninue', default='y', help='if continue to train')
	parser.add_argument('--start_epoch', default=-1, type=int,
	                    help='where to being the epoch, 0 for scratch otherwise all continue')
	parser.add_argument('--init_type', default='xavier', help='weight initialization mode, gain 0.02 fixed in')
	parser.add_argument('--n_thread', default=10, type=int, help='how many threads')
	parser.add_argument('--save_step', default=1, type=int, help='how many steps to save model')
	parser.add_argument('--if_pinMem', action='store_false', help='if pin memory to accelerate. Not working on windows')
	parser.add_argument('--if_finalTest', default='n',
	                    help='if run a final test and keep the result after training session')

	# -- visualization
	parser.add_argument('--display_id', type=int, default=-1, help='window id of the web display')
	parser.add_argument('--display_server', type=str, default="http://localhost",
	                    help='visdom server of the web display')
	parser.add_argument('--display_env', type=str, default='main',
	                    help='visdom display environment name (default is "main")')
	parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
	parser.add_argument('--use_html', action='store_true', help='if use html')
	parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
	parser.add_argument('--display_ncols', type=int, default=3,
	                    help='if positive, display all images in a single visdom web panel with certain number of images per row.')
	parser.add_argument('--update_html_freq', type=int, default=10,
	                    help='frequency of saving training results to html, def 1000 ')
	parser.add_argument('--print_freq', type=int, default=10,
	                    help='frequency of showing training results on console, def 100')
	parser.add_argument('--no_html', action='store_true',
	                    help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

	# -- test setting
	parser.add_argument('--if_test', action='store_true')
	parser.add_argument('--testset', default='SLP',
	                    help='testset, usually single [Human3dM|ScanAva|MuPoTS|SURREAL]')
	parser.add_argument('--testIter', type=int, default=-1,
	                    help='test iterations final and epoch test, -1 for all, DBP')
	parser.add_argument('--yn_flipTest', default='y')
	parser.add_argument('--if_gtRtTest', default='y', help='if use gt distance for root')
	parser.add_argument('--if_adj', default='y', help='if adjust the root location to adapt different dataset')
	parser.add_argument('--testImg', default=None,
	                    help='if indicate image, test will show the skeleton and 2d images of it')
	parser.add_argument('--bone_type', default='A2J',
	                    help='choose the type of joints to be test against  [model|scanava|h36m|cmJoints], what to be trained')
	parser.add_argument('--if_loadPreds', default='n',
	                    help='if load preds in test func, already saved data to accelerate')
	parser.add_argument('--if_test_ckpt', default='n', help='if check intermediate checkpoint')
	parser.add_argument('--svVis_step', default=1, type=int, help='step to save visuals')
	parser.add_argument('--test_par', default='test',
	                    help='the exact test portion, could be [testInLoop|test|train|all], can use the model to test on train set or test set')  # I just save default first

	opts, _ = parser.parse_known_args()  # all cmd infor

	return opts

def aug_opts(opts):
	# base on given opts, add necessary informations
	opts.input_shape = opts.sz_pch[::-1]  # tuple size
	# to update ---
	opts.depth_dim = opts.input_shape[0] // 4  # save as output shape, df 64. similar to intergral
	opts.bbox_3d_shape = (2000, 2000, 2000)  # depth, height, width,
	opts.pixel_mean = (0.485, 0.456, 0.406)  # perhaps for RGB normalization  after divide by 255
	opts.pixel_std = (0.229, 0.224, 0.225)  # for h36m version   remove later
	opts.SLP_fd = os.path.join(opts.ds_fd, 'SLP', opts.SLP_set) # SLP folder [danaLab|simLab]

	covStr = ''
	if 'uncover' in opts.cov_li:
		covStr += 'u'
	if 'cover1' in opts.cov_li:
		covStr += '1'
	if 'cover2' in opts.cov_li:
		covStr += '2'

	modStr = '-'.join(opts.mod_src)
	input_nc = 0
	for mod in opts.mod_src:
		if 'RGB' in mod:
			input_nc+=3
		else:
			input_nc += 1
	opts.input_nc = input_nc

	opts.clipMode = '01'  # for save image purpose
	if not os.path.isabs(opts.output_dir):
		opts.output_dir = os.path.abspath(opts.output_dir)
	nmT = '-'.join(opts.trainset)  # init
	dct_opt = vars(opts)
	# set tne naming needed attirbutes
	suffix_train = (opts.suffix_ptn_train.format(
		**vars(opts))) if opts.suffix_ptn_train != '' else ''  # std pattern
	nmT = '_'.join([nmT, modStr, covStr, suffix_train, opts.suffix_exp_train])  # ds+ ptn_suffix+ exp_suffix
	opts.name = nmT  # current experiment name
	opts.exp_dir = osp.join(opts.output_dir, nmT)
	opts.model_dir = osp.join(opts.exp_dir, 'model_dump')
	opts.vis_dir = osp.join(opts.exp_dir, 'vis', opts.test_par) #during train   vis/[train|test]
	opts.log_dir = osp.join(opts.exp_dir, 'log')
	opts.rst_dir = osp.join(opts.exp_dir, 'result')
	opts.num_gpus = len(opts.gpu_ids)
	opts.web_dir = osp.join(opts.exp_dir, 'web')
	vis_sub = opts.testset
	if 'SLP' == opts.testset:
		vis_sub = '_'.join(['SLP', opts.SLP_set])     # SLP add split to it
	opts.vis_test_dir = osp.join(opts.vis_dir, vis_sub)  # specific test dataset
	opts.batch_size = opts.batch_size_pGPU * opts.num_gpus  # the actual batch size

	yn_dict = {'y': True, 'n': False}
	opts.if_flipTest = yn_dict[opts.yn_flipTest]
	opts.use_gt_info = yn_dict[opts.if_gtRtTest]

	# test name needs start_epoch
	sfx_test = (opts.suffix_ptn_test.format(**vars(opts))) if opts.suffix_ptn_test != '' else ''
	opts.nmTest = '_'.join((sfx_test, opts.suffix_exp_test))
	# otherwise, do nothing use the current start_epoch
	return opts


def print_options(opt, if_sv=False):
	"""Print and save options

	It will print both current options and default values(if different).
	It will save options into a text file / [checkpoints_dir] / opt.txt
	"""
	message = ''
	message += '----------------- Options ---------------\n'
	for k, v in sorted(vars(opt).items()):
		comment = ''
		# default = self.parser.get_default(k)
		# if v != default:
		# 	comment = '\t[default: %s]' % str(default)
		message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
	message += '----------------- End -------------------'
	print(message)

	# save to the disk
	if if_sv:
		ut.make_folder(opt.exp_dir)  # all option will mk dirs  # saved to json file in set_env
		file_name = os.path.join(opt.exp_dir, 'opts.txt'.format(
			opt.start_epoch))  #
		with open(file_name, 'wt') as opt_file:
			opt_file.write(message)
			opt_file.write('\n')

# opts = parseArgs()

def set_env(opts):  # to be changed accordingly for rst fd
	# set sys paths
	# sys.path.insert(0, 'common')  # not using commong
	from utils.utils import add_pypath, make_folder
	add_pypath(osp.join('data'))        # actually we can use ds directly
	for i in range(len(opts.trainset)):
		add_pypath(osp.join('data', opts.trainset[i]))
	# if opts.cocoapi_dir:
	# 	add_pypath(opts.cocoapi_dir)  # add coco dir to it
	add_pypath(osp.join('data', opts.testset))

	# add folders
	make_folder(opts.model_dir)
	make_folder(opts.vis_dir)
	make_folder(opts.log_dir)
	make_folder(opts.rst_dir)
	make_folder(opts.web_dir)

