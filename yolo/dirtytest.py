import tensorflow as tf
import numpy as np
import cv2
import yolo_netfactory as nf
import random_batch as rb
import YOLO_tiny_tf
import voc_utils as voc
import os
import utility as ut
from bs4 import BeautifulSoup as soup
import model_utility as mut
import time
import model_utility as mu
import matplotlib.pyplot as plt

key_pairs = {
            'conv1w':'Variable',
            'conv1b':'Variable_1',
            'conv2w':'Variable_2',
            'conv2b':'Variable_3',
            'conv3w':'Variable_4',
            'conv3b':'Variable_5',
            'conv4w':'Variable_6',
            'conv4b':'Variable_7',
            'conv5w':'Variable_8',
            'conv5b':'Variable_9',
            'conv6w':'Variable_10',
            'conv6b':'Variable_11',
            'conv7w':'Variable_12',
            'conv7b':'Variable_13',
            'conv8w':'Variable_14',
            'conv8b':'Variable_15',
            'conv9w':'Variable_16',
            'conv9b':'Variable_17',
            'fc10w':'Variable_18',
            'fc10b':'Variable_19',
            'fc11w':'Variable_20',
            'fc11b':'Variable_21',
            'fc12w':'Variable_22',
            'fc12b':'Variable_23',
            
            }

def recursive_create_var(scope, Nlayers, reduce_percent, init_layers):
    
    scope_dict = {}
    
    for idx in range(Nlayers):
        
        scope_name = scope + '_' + str(idx)
        with tf.variable_scope(scope_name):
            
            name_dict = []
            for i in range(len(init_layers)):

                shape = init_layers[i][1]
                
                if idx >= 1: 
                                       
                    if i > 0 and len(shape)  > 1 : shape[-2] = int(init_layers[i-1][1][-1])
                    
                    shape[-1] = int(init_layers[i][1][-1] - reduce_percent*(init_layers[i][1][-1]))
                    
                    
                init_layers[i][1] = shape
                
                if init_layers[i][0] == 'fc12w':  init_layers[i][1][-1] = 1470
                if init_layers[i][0] == 'fc12b':  init_layers[i][1][-1] = 1470          
                if init_layers[i][0] == 'fc10w':
                    if idx == 0:
                        init_layers[i][1][-2] = 50176
                    if idx == 1:
                        init_layers[i][1][-2] = 40131
                   
          
                tf.get_variable(init_layers[i][0],init_layers[i][1], initializer=tf.contrib.layers.xavier_initializer())

                name_dict.append(init_layers[i][0])
        scope_dict[scope_name] = name_dict 

    return scope_dict


img_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2007/Test/JPEGImages'

if os.path.isdir(img_root):

    labelfiles = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2007/Test/ImageSets/Main'
else:
    img_root = '/home/dashmoment/workspace/dataset/VOCdevkit/VOC2007/Test/JPEGImages'
    labelfiles = '/home/dashmoment/workspace/dataset/VOCdevkit/VOC2007/Test/ImageSets/Main'

yolo_old = YOLO_tiny_tf.YOLO_TF()

with tf.device('/gpu:1'):
    classes = voc.list_image_sets()
    val_list = voc.imgs_from_category_as_list('', 'test', labelfiles)
    
    model_ticket={'root':'yolo_tiny', 'branch':'vanilla'}
    init_layers = mu.model_zoo(model_ticket)
    var_dict = recursive_create_var('recursive', 2, 0.1, init_layers)
    var_list = var_dict['recursive_0']
    


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config = config) as sess:
    
     sess.run(tf.global_variables_initializer())  

     with tf.name_scope('Weight_sum'):        
            with tf.variable_scope('recursive_0') as scope:
                    scope.reuse_variables()           
                   
                    for var in key_pairs:
                        yolo_var = yolo_old.sess.run(tf.get_default_graph().get_tensor_by_name(key_pairs[var]+':0'))
                        op = tf.assign(tf.get_variable(var), yolo_var)
                        sess.run(op)
                        
                        tensors =  sess.run(tf.get_variable(var))
                        tmp2 = np.array()
                        
                        if (len(np.shape(tensors))) == 4:
                            
                            tensors2 = tensors
                            mean = np.mean(np.mean(tensors, axis=0),axis=0)
                            
                            for i in range(mean.shape[-1]):
                                axis_mean = mean[:,i]
                                sort = np.unravel_index(axis_mean.argsort(axis=None), dims=int(len(axis_mean)*0.9))
                            
                        
                      
                    
                    
        
    
















