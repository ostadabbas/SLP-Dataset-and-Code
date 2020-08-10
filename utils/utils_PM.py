# utils to read PM data
import os
import numpy as np
from skimage import io
from skimage import transform
import math
from math import cos, sin
import matplotlib
import matplotlib.pyplot as plt
import warnings

def getPth(dsFd=r'G:\My Drive\ACLab Shared GDrive\datasetPM\danaLab',idx_subj=1, modality='IR', cov='uncover', idx_frm=1):
    if modality in {'depth','IR','PM','RGB'}:   # simple name are image format
        nmFmt = 'image_{:06}.png'       # read in png
    else:
        nmFmt = '{:06d}.npy'        # or read in npy format
    imgPth = os.path.join(dsFd, '{:05d}'.format(idx_subj), modality, cov, nmFmt.format(idx_frm))
    return imgPth

def getImg_dsPM(dsFd=r'G:\My Drive\ACLab Shared GDrive\datasetPM\danaLab',idx_subj=1, modality='IR', cov='uncover', idx_frm=1):
    '''
    directly get image
    :param dsFd:
    :param idx_subj:
    :param modality:
    :param cov:
    :param idx_frm:
    :return:
    '''
    npy_nmSet = {'depthRaw', 'PMarray'}  # mainly use depth raw and PM array
    if modality in npy_nmSet:       # npy format
        nmFmt = '{:06}.npy'
        # imgPth = os.path.join(dsFd, '{:05d}'.format(idx_subj), modality, cov, nmFmt.format(idx_frm))
        # img = np.load(imgPth)
        readFunc = np.load
    else:
        nmFmt = 'image_{:06d}.png'
        readFunc = io.imread
    imgPth = os.path.join(dsFd, '{:05d}'.format(idx_subj), modality, cov, nmFmt.format(idx_frm))
    img = readFunc(imgPth)  # should be 2d array
    img = np.array(img)
    # if len(img.shape)<3:    # all to 3 dim
    #     img = np.expand_dims(img,-1) # add channel to the last
    return img
def affineImg(img, scale=1,deg=0,  shf=(0,0)):
    '''
    scale, rotate and shift around center, same cropped image will be returned with skimage.transform.warp. use anti-clockwise
    :param img:  suppose to be 2D, or HxWxC format
    :param deg:
    :param shf:
    :param scale:
    :return:
    '''
    h,w = img.shape[:2] #
    c_x = (w+1)/2
    c_y = (h+1)/2
    rad = -math.radians(deg) #
    M_2Cs= np.array([
        [scale, 0, -scale * c_x],
        [0, scale, -scale * c_y],
        [0, 0,  1]
    ])
    M_rt = np.array([
        [cos(rad), -sin(rad), 0],
        [sin(rad), cos(rad), 0],
        [0, 0 ,     1]
    ])
    M_2O = np.array([
        [1, 0, c_x+shf[0]],
        [0, 1,  c_y+shf[1]],
        [0, 0 , 1]
                    ])
    # M= M_2O  * M_2Cs
    #M= np.linalg.multi_dot([M_2O, M_rt, M_2Cs]) # [2,2, no shift part?
    M= M_2O @ M_rt @ M_2Cs
    tsfm = transform.AffineTransform(np.linalg.inv(M))
    img_new = transform.warp(img, tsfm, preserve_range=True)
    return img_new

def getDiff(model, dataset, num_test):
    '''
    from the testing set, get all diff and real matrix for accuracy test
    :param model:
    :param opt_test:
    :return:
    '''
    model.eval()
    diff_li = []
    real_li = []
    for i, data in enumerate(dataset):
        if i == num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        # real_B = model.real_B.squeeze().cpu().numpy()
        # fake_B = model.fake_B.squeeze().cpu().numpy()

        real_B = model.real_B.cpu().numpy()
        fake_B = model.fake_B.cpu().numpy()
        diff_abs = np.abs(real_B - fake_B)
        diff_li.append(diff_abs)
        real_li.append(real_B)
    # diff_dStk = np.dstack(diff_li)
    # real_dStk = np.dstack(real_li)
    diff_bStk = np.concatenate(diff_li)
    real_bStk = np.concatenate(real_li)
    return diff_bStk, real_bStk

def ts2Img(ts_bch, R=1, nm_cm=None, if_bch=True):
    '''
    take first tensor from tensor bach and save it to image format, io will deal will the differences across domain
    :param ts_bch: direct output from model
    :return: the image format with axis changed ( I think io can save different range directly, so not handle here), suppose to be 3 dim one.
    '''
    if if_bch:
        ts = ts_bch[0]
    else:
        ts = ts_bch
    image_numpy = ts.data.cpu().float().numpy()
    if 1 == R:
        image_numpy = image_numpy.clip(0, 1)
    elif 2 == R:  # suppose to be clip11,  -1 to 1   make this also to 0, 1 version
        # image_numpy = image_numpy.clip(-1, 1)
        image_numpy = ((image_numpy + 1) / 2).clip(0, 1)
    else:  # otherwise suppose to be uint8 0 ~ 255
        image_numpy = image_numpy.clip(0, 255)
    if image_numpy.shape[0] == 1:  # grayscale to RGB
        if nm_cm:
            cm = plt.get_cmap(nm_cm)
            image_numpy = cm(image_numpy.transpose([1, 2, 0]))  # 1 x 255 x 4
            # print('after trans', image_numpy.shape)
            image_numpy = image_numpy.squeeze()[..., :3]  # only 3 channels.
            # print('image cut 3 channel', image_numpy.shape)
            image_numpy = image_numpy.transpose([2, 0, 1])
        else:
            image_numpy = np.tile(image_numpy, (3, 1, 1))  # make to RGB format
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return image_numpy  # default float

def getDiff_img(model, dataset, num_test, web_dir, if_saveImg=False, num_imgSv= 500, baseNm ='demo'):
    '''
    loop the function and save the diff and images to web_dir. Rename all the images to demo_{i}_[real\fake]_[A|B].png
    :param model:
    :param opt_test:
    :return: vertically stacked  difference between prediction and real and also the gt.  Result concatenated vertically,which is a very tall array.
    '''
    if_verb = False
    model.eval()
    diff_li = []
    real_li = []
    imgFd = os.path.join(web_dir, 'demoImgs')   # save to demo Images folder
    if if_saveImg:
        if not os.path.exists(imgFd):
            os.mkdir(imgFd)
    pth_realA = os.path.join(imgFd, baseNm + '_real_A_{}.png')
    pth_fakeB = os.path.join(imgFd, baseNm + '_fake_B_{}.png')
    pth_realB = os.path.join(imgFd, baseNm + '_real_B_{}.png')
    for i, data in enumerate(dataset):
        if i == num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run forward and compute visual
        if 0 == i % 100:
            print('{} samples processed'.format(i))

        real_A_im = ts2Img(model.real_A)
        real_B_im = ts2Img(model.real_B)
        fake_B_im = ts2Img(model.fake_B)
        if if_verb:
            print('fake_B, min value is {}, max is {}'.format(fake_B_im.min(), fake_B_im.max()))

        if if_saveImg and i < num_imgSv:    # save controlled number of images from test set
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(pth_realA.format(i), real_A_im)
                io.imsave(pth_fakeB.format(i), fake_B_im)
                io.imsave(pth_realB.format(i), real_B_im)

        real_B = model.real_B.cpu().numpy()
        fake_B = model.fake_B.cpu().numpy()
        diff_abs = np.abs(real_B - fake_B)
        diff_li.append(diff_abs)        # still channel first
        real_li.append(real_B)
    # diff_dStk = np.dstack(diff_li)
    # real_dStk = np.dstack(real_li)
    diff_bStk = np.concatenate(diff_li)
    real_bStk = np.concatenate(real_li)
    return diff_bStk, real_bStk

def genPCS(diff_vStk, real_vStk, x, thresh=0.05):
    '''
    generate the PCS vec against x according to preds diff and real values. Only calculate the interesting sensing according to thresh.
    :param diff_vStk:
    :param real_vStk:
    :param x:
    :param thresh: the threshold for interesting sensing.
    :return:
    '''
    y_li = []
    for i in range(len(x)):
        acc = (diff_vStk[real_vStk > thresh] < x[i]).sum() / diff_vStk[real_vStk > thresh].size
        y_li.append(acc)
    return np.array(y_li)   # into array

def drawPCS(rst_li, legend_li, pltNm, rstRt='./results', rgSt= 0, rgTo=0.1, step=11, thresh=0.05,fmt = 'pdf', idx_bold = -1, titleNm = '', sz_lgd = 18, ver=2):
    '''
    plot the PCS in one plot with given rst name list and legends. PCS plot will be saved to <rstRt>/PCSplots with given <pltNm>.
    :param rst_li: is the result folder list when created.
    :param legend_li:
    :param pltNm:
    :param rstRt:
    :param rgTo:
    :param step:
    :param thresh: control the interesting pressure points
    :param sz_lgd: the size of legend
    :param ver: the version number, in version 1, the diff format is different than version 2. We will mainly use version 2 in future. version 1 is only kept for compatibility in case we need plot from old result
    :return:
    '''
    plt.rc('font', family='Times New Roman')
    # plt.rcParams["font.family"] = "Times New Roman"
    # matplotlib.rc('xtick', labelsize=15)
    # matplotlib.rc('ytick', labelsize=15)
    # matplotlib.rc('axes', labelsize=18, titlesize=15)
    # matplotlib.rc('legend', fontsize=18)

    matplotlib.rc('xtick', labelsize=sz_lgd)        # 22 originally
    matplotlib.rc('ytick', labelsize=sz_lgd)
    matplotlib.rc('axes', labelsize=sz_lgd, titlesize=sz_lgd)
    matplotlib.rc('legend', fontsize=sz_lgd)
    # matplotlib.rc('title', fontsize=18)
    # font = {'family': 'Times New Roman',
    #         'weight': 'normal',
    #         'size': 10}
    # matplotlib.rc('font', family='Times New Roman')
    # matplotlib.rc('font', **font)
    # plt.rcParams["font.family"] = 'Times New Roman'
    # matplotlib.rc('text', usetex = True)

    if not len(rst_li) == len(legend_li):
        print('rst list and legend list can not match')
        return -1

    x = np.linspace(rgSt, rgTo, step)
    # y_li = []
    # for pcs in range(len(x)):

    for i, rstFd in enumerate(rst_li):
        if 'clip11' in rstFd:
            bs_sensing = -1
            rg_sensing = 2
            x_calc = rg_sensing * x
        elif 'clip01' in rstFd:
            bs_sensing = 0
            rg_sensing = 1
            x_calc = rg_sensing * x
        else:
            print('no such pmDsProc, exit1')
            os.exit(-1)
        PM_thresh = bs_sensing + rg_sensing * thresh

        if 2 != ver:
            diffPth = os.path.join(rstRt, rstFd, 'test_latest', 'test_diff.npz')
            dataLd = np.load(diffPth)
            diff_vStk = dataLd['diff_dStk']
            real_vStk = dataLd['real_dStk']
        else:
            diffPth = os.path.join(rstRt, rstFd, 'test_latest', 'test_diffV2.npz')
            dataLd = np.load(diffPth)
            fake_vStk = dataLd['fake_vStk']
            real_vStk = dataLd['real_vStk']
            diff_vStk = np.abs(real_vStk - fake_vStk)

        # gen y_rst  from x list
        y = genPCS(diff_vStk, real_vStk, x_calc, thresh=PM_thresh) *100
        if i == idx_bold:
            plt.plot(x,y, label=legend_li[i], linewidth=3)
        else:
            plt.plot(x, y, label=legend_li[i])
    legd = plt.legend(loc='upper left')
    plt.xlabel('Normalized Threshold')
    plt.ylabel('PCS (%)')
    if titleNm:
        plt.title(titleNm)
    plt.gcf().subplots_adjust(bottom=0.2)  # make some rooms
    plt.gcf().subplots_adjust(left=0.2)  # make some rooms
    # emphasize
    for i, text in enumerate(legd.get_texts()):    # can't set individual font
        # print('text', i)
        if idx_bold == i:
            # print('set', i)
            font = {'family':'Times New Roman',
                    'weight':'bold',
                     'size':sz_lgd
                    }
            fontProp = matplotlib.font_manager.FontProperties(**font)
            # text.set_fontweight('bold')
            text.set_fontproperties(fontProp)
    # save the result
    PCSfd = os.path.join(rstRt, 'PCSplots')
    if not os.path.exists(PCSfd):
        os.mkdir(PCSfd)
    pth_save = os.path.join(rstRt, 'PCSplots', pltNm + '.' + fmt)   # default pdf
    plt.savefig(pth_save)   # there is white margin
    # plt.show()

def drawPCSv2(rst_li, legend_li, pltNm, rstRt='./results', rgSt= 0, rgTo=0.1, step=11, thresh=0.05,fmt = 'pdf', idx_bold = -1, titleNm = '', sz_lgd = 18):
    '''
    plot the PCS in one plot with given rst name list and legends. PCS plot will be saved to <rstRt>/PCSplots with given <pltNm>.
    :param rst_li: is the result folder list when created.
    :param legend_li:
    :param pltNm:
    :param rstRt:
    :param rgTo:
    :param step:
    :param thresh: control the interesting pressure points
    :param sz_lgd: the size of legend
    :return:
    '''
    plt.rc('font', family='Times New Roman')
    # plt.rcParams["font.family"] = "Times New Roman"
    # matplotlib.rc('xtick', labelsize=15)
    # matplotlib.rc('ytick', labelsize=15)
    # matplotlib.rc('axes', labelsize=18, titlesize=15)
    # matplotlib.rc('legend', fontsize=18)

    matplotlib.rc('xtick', labelsize=sz_lgd)        # 22 originally
    matplotlib.rc('ytick', labelsize=sz_lgd)
    matplotlib.rc('axes', labelsize=sz_lgd, titlesize=sz_lgd)
    matplotlib.rc('legend', fontsize=sz_lgd)
    # matplotlib.rc('title', fontsize=18)
    # font = {'family': 'Times New Roman',
    #         'weight': 'normal',
    #         'size': 10}
    # matplotlib.rc('font', family='Times New Roman')
    # matplotlib.rc('font', **font)
    # plt.rcParams["font.family"] = 'Times New Roman'
    # matplotlib.rc('text', usetex = True)

    if not len(rst_li) == len(legend_li):
        print('rst list and legend list can not match')
        return -1

    x = np.linspace(rgSt, rgTo, step)
    # y_li = []
    # for pcs in range(len(x)):

    for i, rstFd in enumerate(rst_li):
        # if 'clip11' in rstFd:
        #     bs_sensing = -1
        #     rg_sensing = 2
        #     x_calc = rg_sensing * x
        # elif 'clip01' in rstFd:
        #     bs_sensing = 0
        #     rg_sensing = 1
        #     x_calc = rg_sensing * x
        # else:
        #     print('no such pmDsProc, exit1')
        #     os.exit(-1)
        if 'pix2pix' in rstFd or 'cycle_gan' in rstFd:
            bs_sensing = - 1
            rg_sensing = 2
            x_calc = rg_sensing * x
        else:
            bs_sensing = 0
            rg_sensing = 1
            x_calc = rg_sensing * x

        PM_thresh = bs_sensing + rg_sensing * thresh

        diffPth = os.path.join(rstRt, rstFd, 'test_latest', 'test_diffV2.npz')
        dataLd = np.load(diffPth)
        fake_vStk = dataLd['fake_vStk']
        real_vStk = dataLd['real_vStk']
        diff_vStk = np.abs(real_vStk - fake_vStk)

        # gen y_rst  from x list
        y = genPCS(diff_vStk, real_vStk, x_calc, thresh=PM_thresh) *100
        if i == idx_bold:
            plt.plot(x,y, label=legend_li[i], linewidth=3)
        else:
            plt.plot(x, y, label=legend_li[i])
    legd = plt.legend(loc='upper left')
    plt.xlabel('Normalized Threshold')
    plt.ylabel('PCS (%)')
    if titleNm:
        plt.title(titleNm)
    plt.gcf().subplots_adjust(bottom=0.2)  # make some rooms
    plt.gcf().subplots_adjust(left=0.2)  # make some rooms
    # emphasize
    for i, text in enumerate(legd.get_texts()):    # can't set individual font
        # print('text', i)
        if idx_bold == i:
            # print('set', i)
            font = {'family':'Times New Roman',
                    'weight':'bold',
                     'size':sz_lgd
                    }
            fontProp = matplotlib.font_manager.FontProperties(**font)
            # text.set_fontweight('bold')
            text.set_fontproperties(fontProp)
    # save the result
    PCSfd = os.path.join(rstRt, 'PCSplots')
    if not os.path.exists(PCSfd):
        os.mkdir(PCSfd)
    pth_save = os.path.join(rstRt, 'PCSplots', pltNm + '.' + fmt)   # default pdf
    plt.savefig(pth_save)   # there is white margin
    # plt.show()
    plt.clf()




