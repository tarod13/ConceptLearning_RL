import cv2
import os

def exists_folder(f_name):
    return os.path.isdir(f_name)

n_test = 126
iter_ = 300
eval_epsds = 1

env_names = ['Hopper-v2']
folder_name = 'HopperTest'
common_path = folder_name + '/' + str(n_test)
specific_path = common_path + '/' + str(iter_)

if not exists_folder(specific_path + '_frames'):
    os.mkdir(specific_path + '_frames')

vidcap = cv2.VideoCapture(specific_path + '.avi')
success, image = vidcap.read()
count = 0
while success and count < 1000*eval_epsds:
  cv2.imwrite(specific_path+'_frames/%d.jpg' % count, image)     # save frame as JPEG file      
  success, image = vidcap.read()
  #print('Read a new frame: ', success)
  count += 1