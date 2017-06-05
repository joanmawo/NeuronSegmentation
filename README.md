# NeuronSegmentation
Final Project for the Computer Vision Course

Neuron Segmentation with the DRIVE CNN

1. Obtainig the data
Execute the file NeuronSegmentation_dataPrep.m to obtain the training and evaluation images, their annotatitions, the networks parameters and its weights.
This script will also perform data augmentation, should you wish to train the network.

2. Evaluating the network
Run eval.py in a caffe enviroment. This will evaluate the network with a single image, found in the Evaluation folder. The qualitative results of this evaluations will be stored in the Results folder.
