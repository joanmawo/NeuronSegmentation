import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc
from PIL import Image
import scipy.io
import os
import scipy
import sys

import numpy as np
import sys
sys.path.append('../')
from params import caffe_root, models_root, data_root ## caffe_root = directorio de caffe 
print caffe_root, models_root, data_root
sys.path.insert(0, caffe_root + 'python')
import caffe #Caffe Lib

# Choose between 'DRIVE', 'RIMONE'
database = 'DRIVE' #'RIMONE' #'COB' #
img_exp = 'NEURONS'
iter_weights = '1060'
experiment = 'exp12'

net_struct = models_root+'/'+database+'/deploy_'+database+'.prototxt'
#net_weights = models_root+'/'+img_exp+'/'+database+'_train_iter_'+iter_weights+'.caffemodel'
#net_weights = models_root+'/'+'RIMONE/DRIU_RIMONE.caffemodel'
#net_weights = models_root+'/'+'DRIVE/DRIU_DRIVE.caffemodel'
#net_weights = models_root+'/'+'COB/DRIU_COB.caffemodel'
#net_weights = models_root+'/NEURONS_RIMONE/'+experiment+'/RIMONE_train_iter_'+iter_weights+'.caffemodel'
net_weights = '0003/RIMONE_train_iter_1060.caffemodel'
data_names = '0003/val.lst'
data_root = 'evaluation/'
save_root = '../Results/'+'/val_'+database+'_'+experiment+'/'

caffe.set_device(1)
caffe.set_mode_gpu()

if not os.path.exists(save_root):
    os.makedirs(save_root)

with open(data_names) as f:
    imnames = f.readlines()
print len(imnames), imnames[0].split()[0] 
img_names = [imnames[n].split()[0] for n in range(len(imnames))]
ann_names = [imnames[n].split()[1] for n in range(len(imnames))]

# load net
net = caffe.Net(net_struct, net_weights, caffe.TRAIN)

i = 1
for image_name in img_names:
    save_name = save_root+image_name.split('/')[-1]
    print 'Processing', i, 'of', len(img_names)
    i += 1

    #Read and preprocess data
    im = Image.open(data_root+image_name)
    in_ = np.array(im, dtype=np.float32)
    in_ = np.repeat(in_[:, :, np.newaxis], 3, axis=2)
    print 'Image shape: ', in_.shape
    #in_ = in_[:,:,::-1] #BGR
    in_ -= np.array((30,32,35)) #Mean substraction
    #in_ -= np.array((24.072,24.072,24.072)) #Mean substraction
    in_ = in_.transpose((2,0,1))
        
    #Reshape data layer
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    #Score the model
    net.forward()
    fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
    print 'Fuse shape: ', fuse.shape
        
    #Save the results
    scipy.misc.imsave(save_name, fuse)


