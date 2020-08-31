from __future__ import print_function, division
import numpy as np
import sys
import dill
import copy
import h5py
import random

from shutil import copyfile

import gc

import cv2
import glob
import os

from utils.anom_UCSDholderv1 import anom_UCSDholder
import sys

bdebug = False

from utils.read_list_from_file import read_list_from_file
from utils.dualprint import  dualprint
from utils.ParamManager import ParamManager

def test_hvad(params):
    # experiment params
    data_str = params.get_value('data_str')
    test_str = params.get_value('test_str')

    resz = params.get_value('resz')
    thresh = params.get_value('thresh')
    folder_name = params.get_value('folder_name')

    # frame_feat = 'conv5' # 'raw'

    dataholder = anom_UCSDholder(data_str, resz)

    data_folder = dataholder.data_folder


    feat_folder = '%s/feat' % (data_folder)
    if not os.path.exists(feat_folder):
        os.mkdir(feat_folder)

    model_folder = '%s/model' % (data_folder)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    res_folder = '%s/result' % (data_folder)
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)
    vis_folder = '%s/%s' % (res_folder, folder_name)
    if os.path.isdir(vis_folder) == False:
        os.mkdir(vis_folder)


    imsz = dataholder.imsz
    test_str='test1'
    test_list = read_list_from_file('%s/%s.lst' % (data_folder, test_str))
    thresh = 0.80
    mask_file = '%s/data_mask_org.h5' % (feat_folder)
    #data_mask = None
    data_num = 2550
    if data_str == 'UCSDped1':
        data_num = 6800
    elif data_str == 'avenue':
        data_num = 7670
    data_mask = np.ones((data_num, resz[1], resz[0], 1)) #ped1 6800 ped2 2550 avenue 7670
    skip_frame = 1
    for s in test_list:
        frm_list = []
        frm_folder = '%s/%s' % (data_folder, s)
        _, frm_ext = os.path.splitext(dataholder.img_format)
        frm_files = glob.glob(frm_folder + '/*' + frm_ext)
        frm_files.sort()
        print(frm_files[0])
        for i in range(len(frm_files)):
            filepath,tempfilename = os.path.split(frm_files[i])
            filename,extension = os.path.splitext(tempfilename)
            frm_list.append(int(filename))
        print('frm_list：', frm_list)
        _, testname = s.split('/')
        print('Loading %s' % testname)

        npzfiles = np.load('%s/Test/%s_final.npz' % (vis_folder, testname))
        E_map_final  = npzfiles['Emap_enh']
        npzfiles.close()
        mask1 = (E_map_final < thresh).astype(int)
        mask = np.zeros([len(frm_list), mask1.shape[1], mask1.shape[2]])
        for i in range(len(frm_list)):
            mask[i] = mask1[frm_list[i]]
        print('Before skipping frame:', mask.shape)
        if skip_frame > 1:
            mask = mask[::skip_frame, :, :]
            print('Skipping frame:', mask.shape)
        mask_resz = np.zeros([mask.shape[0], resz[1], resz[0]])
        mask = mask.astype('float32')
        for i in range(mask.shape[0]):
            mask_resz[i, :, :] = cv2.resize(mask[i, :, :], (resz[1], resz[0]))
        mask = mask_resz.astype(int)

        mask = np.expand_dims(mask, axis=3)
        print("mask shape:", mask.shape)
        if data_mask is None:
            data_mask = mask
        else:
            data_mask = np.concatenate([data_mask, mask], axis=0)

    del mask_resz, mask, mask1
    print('data_mask shape', data_mask.shape)

    f = h5py.File(mask_file, 'w')
    f.create_dataset('data', data=data_mask, compression='gzip')
    f.close()
    print('saved to %s' % mask_file)

    del E_map_final
    gc.collect()

    print('Finished.')

if __name__ == "__main__":
    if len(sys.argv)>1:
        # dataset name
        data_str = sys.argv[1]

        # list of layers whose features are used to detect anomaly
        # e.g., "0-1-2" ==> use the low-level feature (0) and the high-level features at the first
        # and second layers
        #       "1" ==> only use the first hidden layer to detect anomalies
        #       "0-2"==> use the low-level feature (0) and the feature at the second hidden layer (2)
        layer_id_str = sys.argv[2]

        # the name of the folder containing trained CAEs and GANs
        cae_folder_name = sys.argv[3]

        # a file contains a list of testing videos
        test_str = sys.argv[4]

    else:
        raise ValueError("Please provide the arguments")

    gan_layer0_folder_name = 'hvad-gan-layer0-v5-brox'
    layer_ids = [int(s) for s in layer_id_str.split('-')]
    print('data_str=%s' % data_str)
    print('layer_ids=',layer_ids)
    print('cae_folder_name = %s' % cae_folder_name)
    print('test_str = %s' % test_str)
    print('use_gt = %d' % use_gt)
    print('gan_layer0_folder_name = %s' % gan_layer0_folder_name)
    params = ParamManager()

    # experiment parameters
    data_range = [-1.0, 1.0]

    net_str = cae_folder_name.split(sep='-lrelu')
    net_str = net_str[0]
    net_str = net_str.replace('hvad-', '')

    # layer = 0
    layer_with_cluster = False

    bh5py = 1
    resz = [256, 256]
    fr_rate_2obj = 0.1 # best for UCSDped1, UCSDped2 and Avenue
    thresh = 0.8

    min_size = 50

    pause_time = 0.0001

    longevity = 30

    # folder_name = 'hvad-net%s-msz%dt%0.3fl%d-broxhier2t-v4-fr2obj%0.2f-ly%s-large-v1a-reshape-beta0.5-comb0.6' % (net_str, min_size,  thresh, longevity, fr_rate_2obj, layer_id_str)
    folder_name = 'hvad-net%s-t%0.3f-ly%s-large-v2-reshape' % (
    net_str, thresh, layer_id_str)
    frame_step = 5

    params.add('cae_folder_name', cae_folder_name, 'hvad')
    params.add('gan_layer0_folder_name', gan_layer0_folder_name, 'hvad')
    params.add('data_str', data_str, 'hvad')
    params.add('test_str', test_str, 'hvad')
    params.add('bh5py', bh5py, 'hvad')
    params.add('frame_step', frame_step, 'hvad')
    params.add('min_size', min_size, 'hvad')
    params.add('data_range', data_range, 'hvad')
    params.add('pause_time', pause_time, 'hvad')
    params.add('longevity', longevity, 'hvad')
    params.add('folder_name', folder_name, 'hvad')

    params.add('layer_ids', layer_ids, 'hvad')

    params.add('resz', resz, 'detector')
    params.add('thresh', thresh, 'detector')
    params.add('fr_rate_2obj', fr_rate_2obj, 'hvad')


    test_hvad(params)