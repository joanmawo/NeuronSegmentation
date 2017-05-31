# Run this function from scripts folder

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from params import caffe_root, models_root ## caffe_root = directorio de caffe 
print caffe_root, models_root
sys.path.insert(0, caffe_root + 'python')
import caffe #Caffe Lib

def write_txtdata(names, labels=None, nametxtfile='name_label.txt'):
    if labels is None:
        text = names
    else:
        text=[names[i]+' {0:.0f}'.format(labels[i]) for i in range(len(names))]
    f = open(nametxtfile,'w')
    [f.write('{}\n'.format(text[n])) for n in range(len(text))]
    f.close()

def write_txtline(line, nametxtfile='name_label.txt'):
    f = open(nametxtfile,'a')
    f.write('{}\n'.format(line))
    f.close()

# helper show filter outputs
def show_filters(net,layer):
    #plt.figure()
    filt_min, filt_max = net.blobs[layer].data.min(), net.blobs[layer].data.max()
    print(filt_min, filt_max, net.blobs[layer].data[0, 1].shape)
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.title("filter {} output".format(i))
        plt.imshow(net.blobs[layer].data[0, i], vmin=filt_min, vmax=filt_max)
        plt.tight_layout()
        #plt.axis('off')
	plt.savefig('filters'+layer+'.png')

database = 'DRIVE' #'COB'# 'RIMONE' #
img_exp = 'NEURONS'
experiment = 'exp14'

net_struct = models_root+database+'/train_val_'+database+'.prototxt'
#net_weights = models_root+'/GOODWEIGHTS/RIMONE_train_iter_150.caffemodel'
net_weights = models_root+database+'/DRIU_'+database+'.caffemodel' 
solver_root = models_root+img_exp+'_'+database+'/solver_'+experiment+'.prototxt'
#models_root+database+'/solver_'+database+'.prototxt'
txt_name = models_root+img_exp+'_'+database+'/'+experiment+'.txt'

if not os.path.exists(models_root+'/'+img_exp+'_'+database+'/'+experiment):
    os.makedirs(models_root+'/'+img_exp+'_'+database+'/'+experiment)

### solve
No_images = 3976
batch_size = 15
iter_size = 1
epoch = np.round(No_images/(batch_size*iter_size))
max_epoch =  4
max_iter = epoch*max_epoch  # EDIT HERE increase to train for longer
test_interval = np.round(max_iter / (max_epoch*5))
snapshot = epoch
base_lr = 0.00000001
stepsize = epoch*5

print 'No_images:', No_images, 'batch_size:', batch_size, 'iter_size:', iter_size
print 'epoch:', epoch, 'max_epoch:', max_epoch, 'max_iter:', max_iter, 'test_interval', test_interval, 'snapshot', snapshot, 'base_lr', base_lr

intro = [net_struct+'\n'+net_weights+'\n'+
		 'No_images: '+str(No_images)+
		 '\nbatch_size: '+str(batch_size)+
		 '\niter_size: '+str(iter_size)+
		 '\nepoch: '+str(epoch)+
		 '\nmax_epoch: '+str(max_epoch)+
		 '\nmax_iter: '+str(max_iter)+
		 '\ntest_interval: '+str(test_interval)+
		 '\nbase_lr: '+str(base_lr)+
		 '\nsnapshot: '+str(snapshot)+'\n'
         '\nIter Loss']
write_txtdata(intro, nametxtfile=txt_name)


solver_text = [ '# Network definition'
                '\nnet: "'+net_struct+'"'+
                '\n# No_images: '+str(No_images)+
                '\n# batch_size: '+str(batch_size)+
                '\niter_size: '+str(iter_size)+
                '\n# epoch: '+str(epoch)+
                '\n# max_epoch: '+str(max_epoch)+
                '\nmax_iter: '+str(max_iter)+
                '\ndisplay: 1'+
                '\n# test_interval: '+str(test_interval)+
                '\nbase_lr: '+str(base_lr)+
                '\nlr_policy: "step"'+
                '\ngamma: 0.1'+
                '\nstepsize: '+str(stepsize)+
                '\nmomentum: 0.9'+
                '\nweight_decay: 0.0002'+
                '\nsnapshot: '+str(snapshot)+
                '\nsnapshot_prefix: "../Models/NEURONS_'+database+'/'+experiment+'/RIMONE_train"'
                '\ntest_iter: 0'
                '\ntest_interval: 1000000']
write_txtdata(solver_text, nametxtfile=solver_root)

caffe.set_device(1)
caffe.set_mode_gpu()

solver = []
solver = caffe.SGDSolver(solver_root)

# copy base weights for fine-tuning
solver.net.copy_from(net_weights)

print 'Shape of layers net:'
for layer_name, blob in solver.net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape) 

print 'loss shape: ', solver.net.blobs['fuse_loss'].data

# losses will also be stored in the log
train_loss = np.zeros(max_iter+1)

# the main solver loop
for it in range(max_iter+1): 
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['fuse_loss'].data

    if it == -1:# or it == 2 or it == 3: 
    	print 'data'
    	#print solver.net.blobs['data'].data
    	#show_filters(solver.net,'data')
    	#print 'label'
    	#print solver.net.blobs['label'].data
    	#show_filters(solver.net,'label')
    	print 'conv1_1'
    	print solver.net.blobs['conv1_1'].data
    	show_filters(solver.net,'conv1_1')
    	#print 'conv2_1'
    	#print solver.net.blobs['conv2_1'].data
    	#show_filters(solver.net,'conv2_1')
    	#print 'conv3_1'
    	#print solver.net.blobs['conv3_1'].data
        #show_filters(solver.net,'conv3_1')
    	print 'conv4_1'
    	print solver.net.blobs['conv4_1'].data
        show_filters(solver.net,'conv4_1')
    	#print 'conv5_1'
    	#print solver.net.blobs['conv5_1'].data

        #print 'upside-multi2'
        #print solver.net.blobs['upside-multi2'].data
        #show_filters(solver.net,'upside-multi2')
        #print 'upside-multi3'
        #print solver.net.blobs['upside-multi3'].data
        #show_filters(solver.net,'upside-multi3')
        #print 'upside-multi4'
        #print solver.net.blobs['upside-multi4'].data
        #show_filters(solver.net,'upside-multi4')
        print 'concat-upscore'
        print solver.net.blobs['concat-upscore'].data
        show_filters(solver.net,'concat-upscore')
        print 'fuse_loss'
        print solver.net.blobs['fuse_loss'].data
        #show_filters(solver.net,'fuse_loss')

    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    if it % test_interval == 0:
    	write_txtline(str(it)+'\t'+' {0:.0f}'.format(train_loss[it]), nametxtfile=txt_name)

