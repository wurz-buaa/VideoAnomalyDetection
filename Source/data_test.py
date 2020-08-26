import cv2
import glob
import os
import numpy as np
from PIL import Image
import scipy.io as sio
'''
folder = '/home/mowhite/Mounted/Documents/MyDocuments/Projects/vad_gan/source/data/shanghaitech/training/videos'
output_folder = '/home/mowhite/Mounted/Documents/MyDocuments/Projects/vad_gan/source/data/shanghaitech/training'
ext = 'avi'
img_files = glob.glob("%s/*.%s" % (folder, ext))
img_files.sort()

N = len(img_files)
#video_name = img_files[0]
#print(os.path.splitext(os.path.split(video_name)[-1])[0])
output_file_format = '%s/%s/%04d.jpg'

for i in range(N):
    video_name = img_files[i]
    img_folder = output_folder+'/'+os.path.splitext(os.path.split(video_name)[-1])[0]
    vc = cv2.VideoCapture(video_name)
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
    c = 1
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        output_file = output_file_format % (output_folder, os.path.splitext(os.path.split(video_name)[-1])[0], c)
        cv2.imwrite(output_file, frame)
        print('saving %s', output_file)
        c=c+1
        cv2.waitKey(1)
        rval, frame = vc.read()
    vc.release()
print('Finished')
'''
'''
vc=cv2.VideoCapture("/home/hqd/PycharmProjects/1/1/19.MOV")
c=1
if vc.isOpened():
    rval,frame=vc.read()
else:
    rval=False
while rval:
    rval,frame=vc.read()
    cv2.imwrite('/home/hqd/PycharmProjects/1/1/19/'+str(c)+'.jpg',frame)
    c=c+1
    cv2.waitKey(1)
vc.release()
'''


folder = '/home/mowhite/Mounted/Documents/MyDocuments/Projects/vad_gan/source/data/UCSD/UCSDped1/Test1'
#folder = '/home/mowhite/Mounted/Documents/MyDocuments/Projects/vad_gan/source/data/UCSD/UCSDped2/Test1'
'''
ext = 'mat'
mask_files = glob.glob("%s/*.%s" % (folder, ext))
mask_files.sort()
output_file_format = '%s/%04d.bmp'
N = len(mask_files)
for i in range(N):
    mask_name = mask_files[i]
    mask_folder = folder+'/'+os.path.splitext(os.path.split(mask_name)[-1])[0]+'_gt'
    #mask = np.load(mask_name)
    maskfile = sio.loadmat(mask_name)
    mask = maskfile['volLabel']
    #mask.astype(int)
    print(mask[0].shape, mask[0].dtype)
    mask[0] = mask[0] * 255
    if not os.path.exists(mask_folder):
        os.mkdir(mask_folder)
    for j in range(mask[0].shape[0]):
        output_file = output_file_format % (mask_folder, j)
        cv2.imwrite(output_file, mask[0][j])
        print('save to', output_file)
print('Finished')
'''
#'''
files = glob.glob("%s/*" % folder)
files.sort()
N = len(files)
output = open('/home/mowhite/Mounted/Documents/MyDocuments/Projects/vad_gan/source/data/UCSD/UCSDped1/test1.lst', 'w')
for i in range(N):
    output.write('Test1/'+os.path.splitext(os.path.split(files[i])[-1])[0]+'\n')
output.close()
print('Finished')
#'''