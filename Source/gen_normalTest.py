from __future__ import print_function, division
import numpy as np
import sys
import dill
import copy

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
    resz = [256, 256]

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


    imsz = dataholder.imsz

    test_list = read_list_from_file('%s/%s.lst' % (data_folder, test_str))


    vis_folder = '%s/%s' % (res_folder, folder_name)

    if os.path.isdir(vis_folder) == False:
        os.mkdir(vis_folder)

    res_filename = '%s/resfile_normal.txt' %(vis_folder)
    resfile= open(res_filename, 'w')

    thresh = 0.80
    score_thresh = 0.85
    minframeNum = 10
    a = 2
    #tpr_thresh = 0.49
    total_tpr=[]
    total_frame = 0
    total_select = 0
    total_pos = 0
    for s in test_list:
        print('Loading %s' % s)
        npzfiles = np.load('%s/%s_final.npz' % (vis_folder, s))
        E_map_final  = npzfiles['Emap_enh']
        npzfiles.close()
        mask = (E_map_final >= thresh).astype(int)

        print('%s: Loading ground-truth' % s)
        frm_folder = '%s/%s_gt' % (data_folder, s)
        _, gt_ext = os.path.splitext(dataholder.gt_format)
        gt_files = glob.glob(frm_folder + '/*' + gt_ext)
        gt_files.sort()
        gt_listi = []
        gt_list = []
        for (j, file) in enumerate(gt_files):
            img = cv2.imread(file, 0)
            im_resz = cv2.resize(img, (imsz[1], imsz[0]),
                                    interpolation=cv2.INTER_NEAREST)
            im_resz = im_resz / 255.0
            gt_listi.append(im_resz)
        gt_list.append(gt_listi)
        print('-->%d frames' % len(gt_listi))
        GT = np.concatenate(gt_list, axis=0)
        print("GT shape： ",GT.shape)
        total_frame += GT.shape[0]
        
        gt_frame = GT.sum(axis=(1,2))
        gt_frame_bool = gt_frame > 0
        #I = (mask + GT)
        #I_intersect = I >= 2
        #I_frame = I_intersect.sum(axis=(1, 2))
        mask_frame = mask.sum(axis=(1,2))
        score = E_map_final*mask
        score_frame = score.sum(axis=(1,2))
        flag = 0
        normal_start = []
        normal_end = []
        E_map_score = []
        for i in range(GT.shape[0]):
            #score = (E_map_final[i] - E_map_final[i].min()) / E_map_final[i].max()
            if mask_frame[i] == 0:
                E_map_score.append(0)
            else:
                E_map_score.append(score_frame[i]/mask_frame[i])

        E_map_score1 = np.array(E_map_score)
        #E_map_score1 = (E_map_score1 - E_map_score1.min())/E_map_score1.max()
        E_map_score1 = E_map_score1 /E_map_score1.max()
        E_map_score = E_map_score1.tolist()
        flag = 0
        for i in range(GT.shape[0]):
            '''
            if E_map_score[i] == score_thresh and gt_frame_bool[i]==False and flag == 0:
                normal_start.append(i)
                flag = 1
            if (E_map_score[i] > score_thresh or gt_frame_bool[i]==True) and flag == 1:
                if i - normal_start[-1] < minframeNum:
                    del normal_start[-1]
                else:
                    normal_end.append(i-1)
                flag = 0
            '''
            if E_map_score[i] < score_thresh and flag == 0:
                normal_start.append(i)
                flag = 1
            if E_map_score[i] >= score_thresh and flag == 1:
                if i - normal_start[-1] < minframeNum:
                    del normal_start[-1]
                else:
                    normal_end.append(i-1)
                flag = 0

        if flag == 1:
            if i - normal_start[-1] < minframeNum:
                del normal_start[-1]
            else:
                normal_end.append(i-1)
        #'''
        normal_data = None
        for i in range(len(normal_start)):
            data = E_map_final[normal_start[i]:normal_end[i]+1]
            if normal_data is None:
                normal_data = data
            else:
                normal_data = np.concatenate([normal_data, data], axis=0)
        if normal_data is not None:
            mask1 = (normal_data >= 0.02).astype(int)
            score1 = normal_data*mask1
            mask1_frame = mask1.sum(axis=(1,2))
            score1_frame = score1.sum(axis=(1,2))
            #normal_data = normal_data/ normal_data.max()
            #new_thresh = normal_data.mean()
            normal_score = []
            for i in range(normal_data.shape[0]):
                if mask1_frame[i] == 0:
                    normal_score.append(0)
                else:
                    normal_score.append(score1_frame[i]/mask1_frame[i])
            normal_score1 = np.array(normal_score)
            new_thresh = normal_score1.mean()
            normal_score1 = normal_score1/normal_score1.max()
            score_thresh1 = normal_score1.mean()
            score_thresh1 = a*score_thresh1*(1-score_thresh1)
        else:
            new_thresh = 0
            score_thresh1 = 0
        new_thresh = float('%.2f' % new_thresh)
        score_thresh1 = float('%.2f' % score_thresh1)
        mask = (E_map_final >= new_thresh).astype(int)
        score = E_map_final*mask
        mask_frame = mask.sum(axis=(1,2))
        score_frame = score.sum(axis=(1,2))

        normal_start = []
        normal_end = []
        E_map_score = []
        for i in range(GT.shape[0]):
            #score = (E_map_final[i] - E_map_final[i].min()) / E_map_final[i].max()
            if mask_frame[i] == 0:
                E_map_score.append(0)
            else:
                E_map_score.append(score_frame[i]/mask_frame[i])

        E_map_score1 = np.array(E_map_score)
        #E_map_score1 = (E_map_score1 - E_map_score1.min())/E_map_score1.max()
        E_map_score1 = E_map_score1 /E_map_score1.max()
        E_map_score = E_map_score1.tolist()
        flag = 0
        for i in range(GT.shape[0]):
            if E_map_score[i] < score_thresh1 and flag == 0:
                normal_start.append(i)
                flag = 1
            if E_map_score[i] >= score_thresh1 and flag == 1:
                if i - normal_start[-1] < minframeNum:
                    del normal_start[-1]
                else:
                    normal_end.append(i-1)
                flag = 0

        if flag == 1:
            if i - normal_start[-1] < minframeNum:
                del normal_start[-1]
            else:
                normal_end.append(i-1)
        print("normal_start: ", normal_start)
        print("normal_end: ", normal_end)
        #'''
        #tpr_frame1 = []
        if len(normal_start) > 0:
            resfile.write(s)
            resfile.write('：\n')
        for i in range(len(normal_start)):
            pos = 0
            total = normal_end[i] - normal_start[i] + 1
            for j in range(normal_start[i], normal_end[i]+1):
                if gt_frame_bool[j] == False:
                    pos += 1
            tpr_frame = pos / total
            total_tpr.append(tpr_frame)
            total_select += total
            total_pos += pos
            resfile.write('frame：'+str(normal_start[i])+'-'+str(normal_end[i])+'(tpr_frame：'+str(float('%.2f' %tpr_frame))+')')
            resfile.write('\n')
            #tpr_frame1.append(tpr_frame)
        
        copy = 0
        if copy == 1:
            if len(normal_start) > 0:
                print('%s: copy files' % s)
                frm_folder = '%s/%s' % (data_folder, s)
                _, frm_ext = os.path.splitext(dataholder.img_format)
                frm_files = glob.glob(frm_folder + '/*' + frm_ext)
                frm_files.sort()
                target_frmfolder = '%s/newdata/%s' % (data_folder, s)
                if not os.path.exists(target_frmfolder):
                    os.makedirs(target_frmfolder)
                
                for i in range(len(normal_start)):
                    for j in range(normal_start[i], normal_end[i]+1):
                        target_frmfile = '%s/%03d%s' % (target_frmfolder, j, frm_ext)
                        copyfile(frm_files[j], target_frmfile)
                        

    total_tpr_array = np.array(total_tpr)
    resfile.write('tpr_mean：'+str(float('%.2f' %total_tpr_array.mean())))
    resfile.write('\n')
    resfile.write('total_frame:'+str(total_frame)+', total_select:'+str(total_select)+', total_pos:'+str(total_pos))
    print(float('%.2f' %total_tpr_array.mean()))
    print('total_frame:', total_frame)
    print('total_select:', total_select)
    print('total_pos:', total_pos)
    del E_map_final
    gc.collect()
    resfile.close()

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



