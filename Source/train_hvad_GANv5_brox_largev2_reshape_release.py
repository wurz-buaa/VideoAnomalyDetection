from __future__ import print_function, division
import numpy as np
import socket

import dill
import copy
import h5py
import scipy.io as sio
import random

import gc

from utils.util import norm_data
from pix2pix_so_v1_func_largev1a import norm_OF_01


computer_name = socket.gethostname()

# SYSINFO = system_config()
# if SYSINFO['display']==False:
#     import matplotlib
#     matplotlib.use('Agg')
# else:
#     import matplotlib

from matplotlib import pyplot as plt


import cv2

import os
import time
from utils.anom_UCSDholderv1 import anom_UCSDholder
import sys

bdebug = False

from utils.read_list_from_file import read_list_from_file
from utils.dualprint import dualprint
from utils.ParamManager import ParamManager

def reshape_feat(D, h1, w1, c1):
    num_data, h, w, c = D.shape
    if h * w * c != h1 * w1 * c1:
        print('[ERROR] Shapes do not match: (%d, %d, %d) vs (%d, %d, %d)' % (h, w, c, h1, w1, c1))
        return
    D1 = np.zeros((num_data, h1, w1, c1))
    if c1 < c:  # from # large number of channels into smaller one

        n_h = int(h1 / h)
        n_w = int(w1 / w)

        for i in range(n_h):
            for j in range(n_w):
                k = i * n_w + j
                D1[:, i * h:(i + 1) * h, j * w:(j + 1) * w, :] = D[:, :, :, k * c1:(k + 1) * c1]
        return D1

    else:  # from # smaller number of channels into larger one
        n_h = int(h / h1)
        n_w = int(w / w1)
        for i in range(n_h):
            for j in range(n_w):
                k = i * n_w + j
                D1[:, :, :, k * c:(k + 1) * c] = D[:, i * h1:(i + 1) * h1, j * w1:(j + 1) * w1, :]
        return D1

def process_feat_reshape_resize(data_F, resz):
    num_data, height, width, num_c = data_F.shape
    height1, width1, num_c1 = compute_reshape(height, width, num_c)
    print(height1, width1, num_c1, height1 * width1 * num_c1)

    data_F_resz = reshape_feat(data_F, height1, width1, num_c1)

    print('reshape features from (%d, %d, %d) to (%d, %d, %d)' % (data_F.shape[1], data_F.shape[2],
                                                                  data_F.shape[3],
                                                                  data_F_resz.shape[1],
                                                                  data_F_resz.shape[2],
                                                                  data_F_resz.shape[3]))

    data_F_new = np.zeros([data_F_resz.shape[0], resz[0], resz[1], data_F_resz.shape[3]])

    for i in range(data_F_resz.shape[0]):
        data_F_new[i, :, :, :] = cv2.resize(data_F_resz[i, :, :, :], (resz[1], resz[0]))

    print('Resizing to:', data_F_new.shape)
    del data_F_resz
    return data_F_new

def compute_reshape(height, width, num_c):
    height1 = height
    width1 = width
    num_c1 = num_c
    while num_c1 % 4 == 0 and num_c1 >= 32 and height1 < 256 and width1 < 256:
        num_c1 = int(num_c1 / 4)
        height1 *= 2
        width1 *= 2
    return height1, width1, num_c1

def train_hvad_GANv5(params):
    import pix2pix_so_v1_func_largev1a as pix2pix
    # experiment params
    mode = params.get_value('mode')
    data_str = params.get_value('data_str')
    train_str = params.get_value('train_str')

    layers = params.get_value('layers')
    layers_with_clusters = params.get_value('layers_with_clusters')
    cae_folder_name = params.get_value('cae_folder_name')
    skip_frame = params.get_value('skip_frame')
    bshow = params.get_value('bshow')
    bh5py = params.get_value('bh5py')

    resz = params.get_value('resz')
    direction = params.get_value('direction')

    # learner parameters
    lr_rate = params.get_value('lr_rate')
    w_init = params.get_value('w_init')
    num_epochs = params.get_value('num_epochs')
    train_flag = params.get_value('train_flag')

    # a.batch_size = 1

    OF_scale = 0.3
    dataholder = anom_UCSDholder(data_str, resz)

    data_folder = dataholder.data_folder
    feat_folder = '%s/feat' % (data_folder)
    if not os.path.exists(feat_folder):
        os.mkdir(feat_folder)

    model_folder = '%s/model' % (data_folder)
    MODEL_DIR = model_folder
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    res_folder = '%s/result' % (data_folder)
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)
    print('Model folder = %s' % model_folder)

    model_folder_name = 'hvad-gan-layer0-v5-brox'
    hvad_model_folder = '%s/%s' % (model_folder, model_folder_name)
    if os.path.isdir(hvad_model_folder) == False:
        os.mkdir(hvad_model_folder)

    flog = open('%s/log_train_hvad_GANv5_brox_large_v2_reshape.txt' % hvad_model_folder, 'wt')
    imsz = dataholder.imsz

    mask_file_org = '%s/data_mask_org.h5' % (feat_folder)
    mask_file = '%s/data_mask.h5' % (feat_folder)
    F_file = '%s/data_F.h5' % (feat_folder)
    M_file = '%s/data_M.h5' % (feat_folder)

    if resz is None:
        resz = imsz

    # load cluster result
    if np.array(layers_with_clusters).sum() > 0:
        cluster_file = '%s/cluster_scene_8x16/cluster.npz' % model_folder
        npz_data = np.load(cluster_file)
        C = npz_data['C_compact_im']

    if bshow==1:
        fig = plt.figure()

    train_time_start = time.time()
    train_list = read_list_from_file('%s/%s.lst' % (data_folder, train_str))
    for l in range(len(layers)):
        print('[%d] Training GAN on layer %d using with/without mask %d' % (l, layers[l], layers_with_clusters[l]))
        if layers[l] == 0:

            # load features
            frame_file_format = '%s/%s_resz%dx%d_raw_v3.npz'
            OF_file_format = '%s/%s_sz256x256_BroxOF.mat'

            # load all data for training
            F = None
            data_O = None
            for s in train_list:
                # frm_folder = "%s/%s" % (data_folder, s)
                # feat_file = feat_file_format  % (
                # feat_folder, s, resz[0], resz[1], h, w, h_step, w_step, strfeature)
                frame_file = frame_file_format % (feat_folder, s, resz[0], resz[1])
                if os.path.isfile(frame_file):
                    dualprint('Loading %s' % s, flog)
                    F_s = np.load(frame_file)
                    print('data shape:', F_s.shape)
                    if skip_frame > 1:
                        F_s = F_s[::skip_frame, :, :]
                        print('Skipping frame:', F_s.shape)
                    if F is None:
                        F = F_s
                    else:
                        F = np.concatenate([F, F_s], axis=0)
                else:
                    dualprint('File %s doesn''t exists' % frame_file, flog)
                    raise ValueError('File %s doesn''t exists' % frame_file)

            dualprint('Convert frame and optical flow into [-1.0, 1.0]')
            # convert frame data in [0.0, 1.0] to [-1.0, 1.0]
            F = pix2pix.preprocess(F)
            data_F = np.stack((F, F, F), axis=3)
            del F, F_s
            print('data_F min %f max %f' % (data_F.min(), data_F.max()))
            print('data_F shape', data_F.shape)
            data_shape = data_F.shape

            idx = np.arange(data_shape[0])
            seed = random.randint(0, 2**31 - 1)
            np.random.seed(seed)
            random.seed(seed)
            rng_state = np.random.get_state()
            print('before shuffling')
            print('corresponding idx', idx[:10])

            np.random.shuffle(idx)
            print('data_F[:10].mean(): %f' % (data_F[:10, :, :, :].mean()))

            np.random.set_state(rng_state)
            np.random.shuffle(data_F)
            print('after shuffling')
            print('corresponding idx', idx[:10])
            print('data_F[:10].mean(): %f' % (data_F[:10, :, :, :].mean()))
            
            # store features
            #F_file_format = '%s/data_F.npy'
            #M_file_format = '%s/%s_data_M.npz'
            F_file = '%s/data_F.h5' % (feat_folder)
            #np.save(F_file, data_F)
            f = h5py.File(F_file, 'w')
            f.create_dataset('data', data=data_F, compression='gzip')
            f.close()
            print('saved to %s' % F_file)
            del data_F
            
            gc.collect()

            times = 0
            datasize = 0
            M_file = '%s/data_M.h5' % (feat_folder)
            for s in train_list:
                OF_file = OF_file_format % (feat_folder, s)
                if os.path.isfile(OF_file):

                    if bh5py == 1:
                        f_h5py = h5py.File(OF_file, 'r')
                        OF = f_h5py['O']
                        OF = np.array(OF).T
                        print(OF.shape)
                    else:
                        mat_data = sio.loadmat(OF_file)
                        OF = mat_data['O']
                    OF = norm_OF_01(OF, scale=OF_scale)
                    OF = pix2pix.preprocess(OF)
                    last_OF = OF[-1, :, :, :]
                    last_OF = np.reshape(last_OF, [1, *last_OF.shape])
                    OF = np.concatenate([OF, last_OF], axis=0)

                    # print(OF.shape)
                    if skip_frame > 1:
                        OF = OF[::skip_frame, :, :, :]
                        print('after skipping frames')
                        print(OF.shape)
                    '''
                    if data_O is None:
                        data_O = OF
                    else:
                        data_O = np.concatenate([data_O, OF], axis=0)
                    '''
                    #data_O_resz = np.zeros([OF.shape[0], resz[1], resz[0], 3])
                    data_O_resz = OF

                    #for i in range(OF.shape[0]):
                    #    data_O_resz[i, :, :, :] = cv2.resize(OF[i, :, :, :], (resz[1], resz[0]))
                    if times == 0:
                        f = h5py.File(M_file, 'w')
                        f.create_dataset('data', data=data_O_resz, maxshape=(None, data_O_resz.shape[1], data_O_resz.shape[2], data_O_resz.shape[3]), compression='gzip')
                        f.close()
                        print('saved to %s' % M_file)
                    else:
                        f = h5py.File(M_file, 'a')
                        dataset = f['data']
                        dataset.resize([datasize+data_O_resz.shape[0], data_O_resz.shape[1], data_O_resz.shape[2], data_O_resz.shape[3]])
                        dataset[datasize:datasize+data_O_resz.shape[0]] = data_O_resz
                        f.close()
                        print('saved to %s' % M_file)
                    datasize = datasize + data_O_resz.shape[0]
                    times = times+1
                else:
                    dualprint('File %s doesn''t exists' % OF_file, flog)

            del OF, last_OF
            '''
            data_O_resz = np.zeros([data_O.shape[0], resz[1], resz[0], 3])

            for i in range(data_O.shape[0]):
                data_O_resz[i, :, :, :] = cv2.resize(data_O[i, :, :, :], (resz[1], resz[0]))

            del data_O
            '''
            #gc.collect()
            #data_O_resz = norm_OF_01(data_O_resz, scale=OF_scale)
            #data_O_resz = pix2pix.preprocess(data_O_resz)

            f = h5py.File(M_file, 'r')
            #data_M = data_O_resz
            data_M = f['data'][:]
            print('Loading %s' % M_file)
            f.close()
            del data_O_resz
            gc.collect()

            idx = np.arange(data_shape[0])     
            print('before shuffling')
            print('corresponding idx', idx[:10])

            np.random.set_state(rng_state)
            np.random.shuffle(idx)
            print('data_M[:10].mean(): %f' % (data_M[:10, :, :, :].mean()))

            np.random.set_state(rng_state)
            np.random.shuffle(data_M)
            print('after shuffling')
            print('corresponding idx', idx[:10])
            print('data_M[:10].mean(): %f' % (data_M[:10, :, :, :].mean()))
            
            print('data_M min %f max %f' % (data_M.min(), data_M.max()))
            print('data_M shape', data_M.shape)
            
            #np.save(M_file, data_M)
            f = h5py.File(M_file, 'w')
            f.create_dataset('data', data=data_M, compression='gzip')
            f.close()
            print('saved to %s' % M_file)
            del data_M
            gc.collect()

            if train_flag > 0:
                f = h5py.File(mask_file_org, 'r')
                print('Loading %s' % mask_file_org)
                data_mask = f['data'][:]
                f.close()

                idx = np.arange(data_shape[0])     
                print('before shuffling')
                print('corresponding idx', idx[:10])

                np.random.set_state(rng_state)
                np.random.shuffle(idx)
                print('data_mask[:10].mean(): %f' % (data_mask[:10, :, :, :].mean()))

                np.random.set_state(rng_state)
                np.random.shuffle(data_mask)
                print('after shuffling')
                print('corresponding idx', idx[:10])
                print('data_mask[:10].mean(): %f' % (data_mask[:10, :, :, :].mean()))
                
                print('data_mask min %f max %f' % (data_mask.min(), data_mask.max()))
                print('data_mask shape', data_mask.shape)
                
                f = h5py.File(mask_file, 'w')
                f.create_dataset('data', data=data_mask, compression='gzip')
                f.close()
                print('saved to %s' % mask_file)
                del data_mask
                gc.collect()

        else:
            # load extracted high level features.
            #cae_folder = '%s/%s' % (model_folder, cae_folder_name)
            #cae_folder_name = None
            if cae_folder_name is not None:
                cae_folder = '%s/%s' % (model_folder, cae_folder_name)
                # loading data
                HF = None
                HM = None
                times = 0
                datasize = 0
                F_file = '%s/data_F.h5' % (feat_folder)
                for s in train_list:
                    feat_file1 = '%s/%s_resz%dx%d_cae1_layer%d.npz' % (cae_folder, s, resz[0], resz[1], layers[l])

                    if os.path.isfile(feat_file1):
                        dualprint('Loading %s of cae1' % s, flog)
                        npz_data = np.load(feat_file1)
                        HF_s = npz_data['feat']
                        if skip_frame > 1:
                            HF_s = HF_s[::skip_frame, :, :, :]
                            print('Skipping frame:', HF_s.shape)
                        '''
                        if HF is None:
                            HF = HF_s
                        else:
                            HF = np.concatenate([HF, HF_s], axis=0)
                        '''
                        if times == 0:
                            f = h5py.File(F_file, 'w')
                            f.create_dataset('data', data=HF_s, maxshape=(None, HF_s.shape[1], HF_s.shape[2], HF_s.shape[3]), compression='gzip')
                            f.close()
                            print('saved to %s' % F_file)
                        else:
                            f = h5py.File(F_file, 'a')
                            dataset = f['data']
                            dataset.resize([datasize+HF_s.shape[0], HF_s.shape[1], HF_s.shape[2], HF_s.shape[3]])
                            dataset[datasize:datasize+HF_s.shape[0]] = HF_s
                            f.close()
                            print('saved to %s' % F_file)
                        datasize = datasize + HF_s.shape[0]
                        times = times+1
                    else:
                        dualprint('File %s does not exists' % feat_file1, flog)
                        raise ValueError('File %s does not exists' % feat_file1)

                del HF_s
                dualprint('Convert frame and optical flow into [-1.0, 1.0]')
                # convert frame data in [0.0, 1.0] to [-1.0, 1.0]


                f = h5py.File(F_file, 'r')
                HF = f['data'][:]
                print('Loading %s' % F_file)
                f.close()
                print('Before converting to [-1, 1]')
                print('HF min %f max %f' % (HF.min(), HF.max()))
                print('HF shape', HF.shape)

                scale = 0.3

                HF, HF_mean, HF_std = norm_data(HF, shift=0.0, scale=scale)

                HF = np.minimum(np.maximum(HF, -1.0), 1.0)
                print('After converting to [-1, 1]')
                print('HF min %f max %f' % (HF.min(), HF.max()))
                print('HF shape', HF.shape)
                data_F = HF
                del HF
                num_data, height, width, num_c = data_F.shape

                height1, width1, num_c1 = compute_reshape(height, width, num_c)
                # store features
                #F_file_format = '%s/data_F.h5'
                #M_file_format = '%s/%s_data_M.npz'
                
                #np.save(F_file, data_F)
                f = h5py.File(F_file, 'w')
                f.create_dataset('data', data=data_F, compression='gzip')
                print('saved to %s' % F_file)
                f.close()
                del data_F
                gc.collect()

                times = 0
                datasize = 0
                M_file = '%s/data_M.h5' % (feat_folder)
                for s in train_list:
                    feat_file2 = '%s/%s_resz%dx%d_cae2_layer%d.npz' % (
                    cae_folder, s, resz[0], resz[1], layers[l])


                    if os.path.isfile(feat_file2):
                        dualprint('Loading %s of cae2' % s, flog)
                        npz_data = np.load(feat_file2)
                        HM_s = npz_data['feat']
                        if skip_frame > 1:
                            HM_s = HM_s[::skip_frame, :, :, :]
                            print('Skipping frame:', HM_s.shape)
                        '''
                        if HM is None:
                            HM = HM_s
                        else:
                            HM = np.concatenate([HM, HM_s], axis=0)
                        '''
                        if times == 0:
                            f = h5py.File(M_file, 'w')
                            f.create_dataset('data', data=HM_s, maxshape=(None, HM_s.shape[1], HM_s.shape[2], HM_s.shape[3]), compression='gzip')
                            f.close()
                            print('saved to %s' % M_file)
                        else:
                            f = h5py.File(M_file, 'a')
                            dataset = f['data']
                            dataset.resize([datasize+HM_s.shape[0], HM_s.shape[1], HM_s.shape[2], HM_s.shape[3]])
                            dataset[datasize:datasize+HM_s.shape[0]] = HM_s
                            f.close()
                            print('saved to %s' % M_file)
                        datasize = datasize + HM_s.shape[0]
                        times = times+1
                    else:
                        dualprint('File %s does not exists' % feat_file2, flog)
                        raise ValueError('File %s does not exists' % feat_file2)

                del HM_s
                f = h5py.File(M_file, 'r')
                HM = f['data'][:]
                print('Loading %s' % M_file)
                f.close()
                print('HM min %f max %f' % (HM.min(), HM.max()))
                print('HM shape', HM.shape)
                print('After converting to [-1, 1]')
                
                HM, HM_mean, HM_std = norm_data(HM, shift=0.0, scale=scale)
  
                HM = np.minimum(np.maximum(HM, -1.0), 1.0)

                print('HM min %f max %f' % (HM.min(), HM.max()))
                print('HM shape', HM.shape)
                data_M = HM
                del HM
                gc.collect()
                mean_std_file = '%s/mean_std_layer%d_large_v2.dill' % (cae_folder, layers[l])

                dill.dump({'cae1_mean': HF_mean, 'cae1_std': HF_std, 'cae1_scale': scale, 'cae2_mean': HM_mean, 'cae2_std': HM_std, 'cae2_scale': scale}, open(mean_std_file, 'wb'))
                dualprint('Saving the mean and std of the layer %d feature as %s' % (layers[l], mean_std_file))

            else:
                raise ValueError('Please provide the reference to the trained cae folder to train pix2pix using high level feature.')

        if layers[l] == 0:
            output_folder = hvad_model_folder
        else:
            output_folder = cae_folder


        if direction == 'AtoB':

            output_folder_FM = '%s/layer%d-usecluster%d-FtoM-large-v2-reshape' % (
                output_folder, layers[l], layers_with_clusters[l])
            if os.path.isdir(output_folder_FM) == False:
                os.mkdir(output_folder_FM)

            flog2 = open('%s/log.txt' % output_folder_FM, 'wt')
            dualprint('[Layer %d] F->M training...' % layers[l], flog)
        elif direction == 'BtoA':
            output_folder_MF = '%s/layer%d-usecluster%d-MtoF-large-v2-reshape' % (
                output_folder, layers[l], layers_with_clusters[l])
            if os.path.isdir(output_folder_MF) == False:
                os.mkdir(output_folder_MF)
            flog2 = open('%s/log.txt' % output_folder_MF, 'wt')
            dualprint('[Layer %d] M->F training...' % layers[l], flog)

        if layers_with_clusters[l] == False:  # no clustering results

            if layers[l] >0:
                #'''
                # reshape feature maps into 256 x 256 images

                #num_data, height, width, num_c = data_F.shape

                #height1, width1, num_c1 = compute_reshape(height, width, num_c)
                print(height1, width1, num_c1, height1*width1*num_c1)

                data_M_resz = reshape_feat(data_M, height1, width1, num_c1)
                del data_M
                gc.collect()
                M_file = '%s/data_M.h5' % (feat_folder)
                f = h5py.File(M_file, 'w')
                f.create_dataset('data', shape=(data_M_resz.shape[0], resz[0], resz[1], data_M_resz.shape[3]), compression='gzip')
                size1 = int(data_M_resz.shape[0]/2)
                size2 = data_M_resz.shape[0] - size1
                data_M = np.zeros([size1, resz[0], resz[1], data_M_resz.shape[3]])
                for i in range(size1):
                    data_M[i, :, :, :] = cv2.resize(data_M_resz[i, :, :, :], (resz[1], resz[0]))
                    #f['data'][i] = cv2.resize(data_M_resz[i, :, :, :], (resz[1], resz[0]))
                f['data'][0:size1] = data_M
                del data_M
                data_M = np.zeros([size2, resz[0], resz[1], data_M_resz.shape[3]])
                for i in range(size2):
                    data_M[i, :, :, :] = cv2.resize(data_M_resz[size1 + i, :, :, :], (resz[1], resz[0]))
                f['data'][size1:data_M_resz.shape[0]] = data_M
                del data_M_resz, data_M
                data_M = f['data'][:]
                f.close()
                data_shape = data_M.shape
                idx = np.arange(data_shape[0])
                seed = random.randint(0, 2**31 - 1)
                np.random.seed(seed)
                random.seed(seed)
                rng_state = np.random.get_state()
                
                print('before shuffling')
                print('corresponding idx', idx[:10])

                np.random.shuffle(idx)
                print('data_M[:10].mean(): %f' % (data_M[:10, :, :, :].mean()))

                np.random.set_state(rng_state)
                np.random.shuffle(data_M)
                print('after shuffling')
                print('corresponding idx', idx[:10])
                print('data_M[:10].mean(): %f' % (data_M[:10, :, :, :].mean()))
                #M_file_format = '%s/data_M.npy'
                
                #np.save(M_file, data_M)
                f = h5py.File(M_file, 'w')
                f.create_dataset('data', data=data_M, compression='gzip')
                f.close()
                print('saved to %s' % M_file)
                del data_M
                gc.collect()

                F_file = '%s/data_F.h5' % (feat_folder)
                if os.path.isfile(F_file):
                    print('Loading %s' % F_file)
                    #data_F = np.load(F_file)
                    f = h5py.File(F_file, 'r')
                    data_F = f['data'][:]
                    f.close()
                    print('data_F shape:', data_F.shape)
                else:
                    print('File %s doesn''t exists' % F_file)

                data_F_resz = reshape_feat(data_F, height1, width1, num_c1)
                
                dualprint('Reshape features from (%d, %d, %d) to (%d, %d, %d)' % (data_F.shape[1], data_F.shape[2],
                                                                              data_F.shape[3], data_F_resz.shape[1],
                                                                              data_F_resz.shape[2], data_F_resz.shape[3]), flog2)
                reshape_info_file = '%s/reshape_info_layer%d_large_v2.dill'% (output_folder, layers[l])
                shape_info = {'org_shape': [data_F.shape[1], data_F.shape[2], data_F.shape[3]],
                              'reshape': [data_F_resz.shape[1], data_F_resz.shape[2], data_F_resz.shape[3]],
                              'resize': [resz[0], resz[1], data_F_resz.shape[3] ]
                              }
                print(shape_info)
                dill.dump(shape_info, open(reshape_info_file, 'wb'))
                dualprint('Saving shape info as: %s' % reshape_info_file, flog2)
                f = h5py.File(F_file, 'w')
                f.create_dataset('data', shape=(data_F_resz.shape[0], resz[0], resz[1], data_F_resz.shape[3]), compression='gzip')
                data_F = np.zeros([size1, resz[0], resz[1], data_F_resz.shape[3]])
                for i in range(size1):
                    data_F[i, :, :, :] = cv2.resize(data_F_resz[i, :, :, :], (resz[1], resz[0]))
                    #f['data'][i] = cv2.resize(data_F_resz[i, :, :, :], (resz[1], resz[0]))
                f['data'][0:size1] = data_F
                del data_F
                data_F = np.zeros([size2, resz[0], resz[1], data_F_resz.shape[3]])
                for i in range(size2):
                    data_F[i, :, :, :] = cv2.resize(data_F_resz[size1 + i, :, :, :], (resz[1], resz[0]))
                f['data'][size1:data_F_resz.shape[0]] = data_F
                del data_F_resz, data_F
                data_F = f['data'][:]
                f.close()
                dualprint('Resizing to: (%d, %d, %d)'% (data_F.shape[1], data_F.shape[2], data_F.shape[3]), flog2)
                data_shape = data_F.shape
                idx = np.arange(data_shape[0])
                print('before shuffling')
                print('corresponding idx', idx[:10])
                np.random.set_state(rng_state)
                np.random.shuffle(idx)
                print('data_F[:10].mean(): %f' % (data_F[:10, :, :, :].mean()))

                np.random.set_state(rng_state)
                np.random.shuffle(data_F)
                print('after shuffling')
                print('corresponding idx', idx[:10])
                print('data_F[:10].mean(): %f' % (data_F[:10, :, :, :].mean()))
                F_file = '%s/data_F.h5' % (feat_folder)
                #np.save(M_file, data_M)
                f = h5py.File(F_file, 'w')
                f.create_dataset('data', data=data_F, compression='gzip')
                f.close()
                print('saved to %s' % F_file)
                del data_F
                gc.collect()

                if train_flag > 0:
                    f = h5py.File(mask_file_org, 'r')
                    print('Loading %s' % mask_file_org)
                    data_mask = f['data'][:]
                    f.close()

                    idx = np.arange(data_shape[0])     
                    print('before shuffling')
                    print('corresponding idx', idx[:10])

                    np.random.set_state(rng_state)
                    np.random.shuffle(idx)
                    print('data_mask[:10].mean(): %f' % (data_mask[:10, :, :, :].mean()))

                    np.random.set_state(rng_state)
                    np.random.shuffle(data_mask)
                    print('after shuffling')
                    print('corresponding idx', idx[:10])
                    print('data_mask[:10].mean(): %f' % (data_mask[:10, :, :, :].mean()))
                    
                    print('data_mask min %f max %f' % (data_mask.min(), data_mask.max()))
                    print('data_mask shape', data_mask.shape)
                    
                    f = h5py.File(mask_file, 'w')
                    f.create_dataset('data', data=data_mask, compression='gzip')
                    f.close()
                    print('saved to %s' % mask_file)
                    del data_mask
                    gc.collect()
                #'''
                #data_shape = (3604, 256, 256, 8)
                '''
                #M_file = M_file_format % (feat_folder)
                if os.path.isfile(M_file):
                    print('Loading %s' % M_file)
                    #data_M = np.load(M_file)
                    f = h5py.File(M_file, 'r')
                    data_M = f['M_data'][:]
                    f.close()
                    print('data_M shape:', data_M.shape)
                else:
                    print('File %s doesn''t exists' % F_file)
                '''
                #F_file = F_file_format % (feat_folder)
                #np.save(F_file, data_F)
                #print('saved to %s' % F_file)
                #del data_F, data_F_resz
                #gc.collect()
                
                
            '''    
            if layers[l] == 0:
                F_file = '%s/data_F.h5' % (feat_folder)
                if os.path.isfile(F_file):
                    print('Loading %s' % F_file)
                    #data_F = np.load(F_file)
                    f = h5py.File(F_file, 'r')
                    data_F = f['F_data'][:]
                    f.close()
                    print('data_F shape:', data_F.shape)
                else:
                    print('File %s doesn''t exists' % F_file)
            '''
            gen_layers1 = [(256, 64),
                           (128, 128),
                           (64, 256),
                           (32, 512),
                           (16, 512),
                           (8, 512),
                           (4, 512),
                           (2, 512)]

            gen_layers2 = [
                (4, 512, 0.5),
                (8, 512, 0.5),
                (16, 512, 0.5),
                (32, 512, 0.0),
                (64, 256, 0.0),
                (128, 128, 0.0),
                (256, 64, 0.0)]

            gen_layers_specs1 = []
            gen_layers_specs2 = []

            for i in range(len(gen_layers1)):
                if gen_layers1[i][0] == data_shape[1]:
                    num_gen_feats = gen_layers1[i][1]
                    num_dis_feats = num_gen_feats
                    gen_layers_specs1 = [out_dim for _, out_dim in gen_layers1[i + 1:]]
                    break
            print('num_gen_feats = %d' % num_gen_feats)
            print('num_dis_feats = %d' % num_dis_feats)
            print('gen_layers_specs1', gen_layers_specs1)
            for i in range(len(gen_layers2)):
                if gen_layers2[i][0] == data_shape[1]:
                    gen_layers_specs2 = gen_layers2[:i + 1]
                    gen_layers_specs2 = [(out_dim, drop) for _, out_dim, drop in gen_layers_specs2]
                    break
            dis_layers = [
                [256, 64, 2],  # gdf, feat_dims, stride
                [128, 128, 2],
                [64, 256, 2],
                [32, 512, 1],
                [31, 1, 1]
            ]
            if data_shape[1] >= 256:
                strides = [2, 2, 2, 1, 1]
            elif data_shape[1] >= 128:
                strides = [2, 2, 1, 1, 1]
            elif data_shape[1] >= 64:
                strides = [2, 1, 1, 1, 1]
            else:
                strides = [1, 1, 1, 1, 1]

            dis_layers_specs = copy.copy(dis_layers)
            for i in range(len(dis_layers)):
                dis_layers_specs[i][2] = strides[i]
            if data_shape[1] <= 32:
                dis_layers_specs = [[32, 512, 1]]
            else:
                for i in range(len(dis_layers)):
                    if dis_layers[i][0] == data_shape[1]:
                        dis_layers_specs = dis_layers[i:-1]

            if direction == 'AtoB':

                pix2pix.pix2pix_func(mode, F_file, M_file, mask_file, data_shape, data_folder, output_folder_FM,
                                     num_epochs,
                                     'AtoB',
                                     num_gen_feats=num_gen_feats,
                                     num_dis_feats = num_dis_feats,
                                     gen_layers_specs1 = gen_layers_specs1,
                                     gen_layers_specs2= gen_layers_specs2,
                                     dis_layers_specs= dis_layers_specs,
                                     layers= layers,
                                     train_flag=train_flag
                                     )
                flog2.write('Finished.\n')
                flog2.close()

            elif direction == 'BtoA':
                pix2pix.pix2pix_func(mode, M_file, F_file, mask_file, data_shape, data_folder, output_folder_MF, num_epochs,
                                     'BtoA',
                                     num_gen_feats=num_gen_feats,
                                     num_dis_feats=num_dis_feats,
                                     gen_layers_specs1=gen_layers_specs1,
                                     gen_layers_specs2=gen_layers_specs2,
                                     dis_layers_specs=dis_layers_specs,
                                     layers= layers,
                                     train_flag=train_flag
                                     )
                flog2.write('Finished.\n')
                flog2.close()
        elif layers_with_clusters[l] == True:  # using clusters
            print('[Error] Not being implemented yet.')
            pass


        else:
            raise ValueError('Invalid value of layers_with_clusters')


    train_time_end = time.time()
    dualprint('Training time: %f (seconds)' % (train_time_end - train_time_start), flog)
    flog.close()
    print('Finished.')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # dataset name
        data_str = sys.argv[1]

        # mode=0: train GAN_F->M
        # mode=1: train GAN_M->F
        mode = sys.argv[2]

        # the layer on which the GAN is trained
        layers = [int(sys.argv[3])]

        # not used
        layers_with_clusters =[sys.argv[4] in ['True', 'true', 't', '1']]

        # direction to train GAN
        # "AtoB": training GAN_F->M
        # "BtoA": training GAN_M->F
        direction = sys.argv[5]

        # a step to sample frames for training
        # in some large datasets such Avenue, this helps to reduce memory and make training faster
        # in our experiment:
        # UCSD Ped1, Ped2: skip_frame = 1
        # Avenue: skip_frame = 2
        skip_frame = int(sys.argv[6])

        # the foldername containing trained models
        if layers[0] > 0:
            cae_folder_name = sys.argv[7]
        else:
            cae_folder_name = None

        train_flag = int(sys.argv[8])

    print('layers:', layers)
    print('layers_with_clusters:', layers_with_clusters)
    print('direction:', direction)
    print('cae_folder_name:', cae_folder_name)
    print('train_flag:', train_flag)
    params = ParamManager()

    # experiment parameters
    if train_flag == 0:
        train_str = 'train'
    elif train_flag == 1:
        train_str = 'all1'
    if skip_frame <0:
        if data_str in ['Avenue', 'Avenue_sz240x360fr1', 'UCSDped1']:
            skip_frame = 2
        else:
            skip_frame = 1
    print('skip_frame = %d\n' % skip_frame)
    bshow = 1
    bh5py = 1

    params.add('mode', mode, 'hvad')
    params.add('data_str', data_str, 'hvad')
    params.add('train_str', train_str, 'hvad')
    params.add('skip_frame', skip_frame, 'hvad')
    params.add('bshow', bshow, 'hvad')
    params.add('bh5py', bh5py, 'hvad')

    resz = [256, 256]

    params.add('layers', layers, 'hvad')
    params.add('layers_with_clusters', layers_with_clusters, 'hvad')
    params.add('cae_folder_name', cae_folder_name, 'hvad')
    params.add('direction', direction, 'hvad')

    params.add('resz', resz, 'hvad')

    lr_rate = 0.01
    w_init = 0.01
    num_epochs = 10

    params.add('lr_rate', lr_rate, 'hvad')
    params.add('w_init', w_init, 'hvad')
    params.add('num_epochs', num_epochs, 'hvad')
    params.add('train_flag', train_flag, 'hvad')

    train_hvad_GANv5(params)