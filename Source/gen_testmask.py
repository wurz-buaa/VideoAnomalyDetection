import numpy as np
import sys
import h5py
import glob
import os
import cv2
import random
import gc

from utils.read_list_from_file import read_list_from_file
from utils.anom_UCSDholderv1 import anom_UCSDholder
dataset='UCSDped2'
#dataset='avenue'
resz = [256, 256]
dataholder = anom_UCSDholder(dataset, resz)
data_folder= '/home/mowhite/Mounted/Documents/MyDocuments/Projects/vad_gan/source/data/UCSD/UCSDped2'
#data_folder= '/home/mowhite/Mounted/Documents/MyDocuments/Projects/vad_gan/source/data/avenue'
feat_folder = '%s/feat' % (data_folder)
test_str='test1'
test_list = read_list_from_file('%s/%s.lst' % (data_folder, test_str))
vis_folder = '/home/mowhite/Mounted/Documents/MyDocuments/Projects/vad_gan/source/data/UCSD/UCSDped2/result/hvad-net32-16-8-release-t0.800-ly0-3-large-v2-reshape'
#vis_folder = '/home/mowhite/Mounted/Documents/MyDocuments/Projects/vad_gan/source/data/avenue/result/hvad-net32-16-8-release-t0.800-ly0-3-large-v2-reshape'
thresh = 0.80
mask_file = '%s/data_mask_org.h5' % (feat_folder)
#data_mask = None
data_mask = np.ones((2550, resz[1], resz[0], 1)) #ped1 6800 ped2 2550 avenue 7670
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
'''
idx = np.arange(data_mask.shape[0])
seed = random.randint(0, 2**31 - 1)
np.random.seed(seed)
random.seed(seed)
rng_state = np.random.get_state()     
print('before shuffling')
print('corresponding idx', idx[:10])

np.random.shuffle(idx)
print('data_mask[:10].mean(): %f' % (data_mask[:10].mean()))

np.random.set_state(rng_state)
np.random.shuffle(data_mask)
print('after shuffling')
print('corresponding idx', idx[:10])
print('data_mask[:10].mean(): %f' % (data_mask[:10].mean()))

print('data_mask min %f max %f' % (data_mask.min(), data_mask.max()))
'''
print('data_mask shape', data_mask.shape)

f = h5py.File(mask_file, 'w')
f.create_dataset('data', data=data_mask, compression='gzip')
f.close()
print('saved to %s' % mask_file)

del E_map_final
gc.collect()

print('Finished.')