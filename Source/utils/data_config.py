from collections import namedtuple
import socket
#import os
def data_config(data_str):
    print('Loading %s configuration' % data_str)
    DATA = namedtuple('DATA', ['UCSDped2_demo','UCSDped1','UCSDped2', 'shanghaitech', 'avenue'])
    DATASET = DATA(UCSDped2_demo = 'UCSDped2_demo', UCSDped1 = 'UCSDped1', UCSDped2 = 'UCSDped2', shanghaitech = 'shanghaitech', avenue = 'avenue')
    # computer_name = os.environ['COMPUTERNAME']
    computer_name = socket.gethostname()
    dataset_folder = None

    dataset_folder = 'data'
    temp_folder = 'temp'
    exp_folder = 'exp'
    DATAINFO = None

    if  data_str== DATASET.UCSDped2_demo:
        DATAINFO = {'data_folder': '%s/UCSD/UCSDped2_test' % dataset_folder,
                    'image_extension': 'tif', 'train_folder_format': 'Train%03d',
                    'test_folder_format': 'Test%03d', 'num_train_videos': 16,
                    'num_test_videos': 12, 'imsz':(240, 360),
                    'threshold': 0.01, 'train_str':'train', 'val_str':'val',
                    'test_str':'test', 'gt_format':'%03d.bmp', 'img_format':'%03d.tif'}
    elif data_str== DATASET.UCSDped1:
        DATAINFO = {'data_folder': '%s/UCSD/UCSDped1' % dataset_folder,
                    'image_extension': 'tif', 'train_folder_format': 'Train%03d',
                    'test_folder_format': 'Test%03d', 'num_train_videos': 34,
                    'num_test_videos': 36, 'imsz':(158, 238),
                    'threshold': 0.01, 'train_str':'train', 'val_str':'val',
                    'test_str':'test', 'gt_format':'%03d.bmp', 'img_format':'%03d.tif'}
    elif data_str== DATASET.UCSDped2:
        DATAINFO = {'data_folder': '%s/UCSD/UCSDped2' % dataset_folder,
                    'image_extension': 'tif', 'train_folder_format': 'Train%03d',
                    'test_folder_format': 'Test%03d', 'num_train_videos': 16,
                    'num_test_videos': 12, 'imsz':(240, 360),
                    'threshold': 0.01, 'train_str':'train', 'val_str':'val',
                    'test_str':'test', 'gt_format':'%03d.bmp', 'img_format':'%03d.tif'}
    elif data_str== DATASET.shanghaitech:
        DATAINFO = {'data_folder': '%s/shanghaitech' % dataset_folder,
                    'image_extension': 'jpg', 'train_folder_format': '%02d_%03d',
                    'test_folder_format': '%02d_%04d', 'num_train_videos': 330,
                    'num_test_videos': 107, 'imsz':(480, 856),
                    'threshold': 0.01, 'train_str':'train', 'val_str':'val',
                    'test_str':'test', 'gt_format':'%03d.bmp', 'img_format':'%04d.jpg'}
    elif data_str== DATASET.avenue:
        DATAINFO = {'data_folder': '%s/avenue' % dataset_folder,
                    'image_extension': 'jpg', 'train_folder_format': '%02d',
                    'test_folder_format': '%02d', 'num_train_videos': 16,
                    'num_test_videos': 21, 'imsz':(360, 640),
                    'threshold': 0.01, 'train_str':'train', 'val_str':'val',
                    'test_str':'test', 'gt_format':'%04d.bmp', 'img_format':'%04d.jpg'}
    DATAINFO['temp_folder'] = temp_folder
    DATAINFO['exp_folder'] = exp_folder
    return DATAINFO
