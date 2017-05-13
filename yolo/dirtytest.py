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


def weight_pruning(yolo_obj, sess, key_pairs, var_list):
    
    for var in var_list:
        tensors = yolo_old.sess.run(tf.get_default_graph().get_tensor_by_name(key_pairs[var]+':0'))
        
        with tf.variable_scope('recursive_1') as scope:
            scope.reuse_variables() 
            tensors_s =  sess.run(tf.get_variable(var))
              
        if np.shape(tensors) != np.shape(tensors_s):
                        
                if (len(np.shape(tensors))) > 1:
                
                    if (len(np.shape(tensors))) > 2:                         
                        dim_sum = np.sum(np.sum(np.sum(tensors, axis=0),axis=0), axis=1)
                    else:
                        dim_sum = np.sum(tensors, axis=1)
                    
                    dim_sum = np.abs(dim_sum)
                    dim_sort = np.unravel_index(dim_sum.argsort(axis=None), dims=int(len(dim_sum)))
                    del_list = [dim_sort[0][x] for x in range(np.shape(tensors)[-2] -  np.shape(tensors_s)[-2])]
                    
                    dim_array = np.delete(tensors,del_list, -2)
                    
                    if (len(np.shape(tensors))) > 2:        
                        kernel_sum = np.sum(np.sum(np.sum(dim_array,axis=0), axis=0),axis=0)
                    else:
                        kernel_sum = np.sum(dim_array, axis=0)
                    
                    kernel_sum = np.abs(kernel_sum)
                    kernel_sort = np.unravel_index(kernel_sum.argsort(axis=None), dims=int(len(kernel_sum)))
                    kdel_list = [kernel_sort[0][x] for x in range(np.shape(tensors)[-1] -  np.shape(tensors_s)[-1])]
                    kernel_array = np.delete(dim_array, kdel_list, -1)
                
                else:
                    kernel_array = np.delete(tensors, kdel_list,0)
                
        else:
            kernel_array = tensors

    
        with tf.variable_scope('recursive_1') as scope:
                scope.reuse_variables() 
                op = tf.assign(tf.get_variable(var), kernel_array)
                sess.run(op)
                tensors_s =  sess.run(tf.get_variable(var))
            
            
    return kernel_array

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
                    
                    shape[-1] = int(init_layers[i][1][-1]*reduce_percent)
                    
                    
                init_layers[i][1] = shape
                
                if init_layers[i][0] == 'fc12w':  init_layers[i][1][-1] = 1470
                if init_layers[i][0] == 'fc12b':  init_layers[i][1][-1] = 1470          
                if init_layers[i][0] == 'fc10w':
                    if idx == 0:
                        init_layers[i][1][-2] = 50176
                    if idx == 1:
                        init_layers[i][1][-2] = 45129
                   
          
                tf.get_variable(init_layers[i][0],init_layers[i][1], initializer=tf.contrib.layers.xavier_initializer())

                name_dict.append(init_layers[i][0])
        scope_dict[scope_name] = name_dict 

    return scope_dict


img_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2007/Test/JPEGImages'
shrink_ratio = 0.9

if os.path.isdir(img_root):

    labelfiles = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2007/Test/ImageSets/Main'
else:
    img_root = '/home/dashmoment/workspace/dataset/VOCdevkit/VOC2007/Test/JPEGImages'
    labelfiles = '/home/dashmoment/workspace/dataset/VOCdevkit/VOC2007/Test/ImageSets/Main'

yolo_old = YOLO_tiny_tf.YOLO_TF()

with tf.device('/gpu:0'):
    classes = voc.list_image_sets()
    val_list = voc.imgs_from_category_as_list('', 'test', labelfiles)
    
    model_ticket={'root':'yolo_tiny', 'branch':'vanilla'}
    init_layers = mu.model_zoo(model_ticket)
    var_dict = recursive_create_var('recursive', 2, shrink_ratio, init_layers)
    var_list = var_dict['recursive_0']
    
x = tf.placeholder(tf.float32,(None,448,448,3))
label = tf.placeholder(tf.float32,(None,1470), name='labels')
keep_prob = tf.placeholder(tf.float32)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config = config) as sess:
    
     sess.run(tf.global_variables_initializer())  
     
     res = weight_pruning(yolo_old, sess, key_pairs, var_list)


     fpath = '/home/dashmoment/workspace/dataset/VOCdevkit/VOC2007/Test/JPEGImages/000011.jpg'    
     w,h,inputs = ut.vocimg_preprocess(fpath)
     src = cv2.imread(fpath)
     
     yolo_ds = nf.glosso_train("recursive_1", 'test', x, var_dict, keep_prob, False)  
     
     for i in range(10):
    
         
         
         s = time.clock()
         prob_label_old = yolo_old.sess.run(yolo_old.fc_19, feed_dict={yolo_old.x:inputs})
         e = time.clock()    
         print("Time old:{}".format(e-s))
         
         s = time.clock()     
         prob_label = sess.run(yolo_ds,feed_dict={x:inputs, keep_prob:1})
         e = time.clock()
         print("Time:{}".format(e-s))
     
     
    
     
     
     
#     lossTicket = {'loss':'yolo_kl_l1'}
#     loss_pair = {'prob':prob_label, 'gloss':0}  
#     loss = mu.loss_zoo(lossTicket, loss_pair, label)
#     
#     cost = sess.run(loss, feed_dict={x:inputs, label:prob_label_old, keep_prob:1})
#     
#     
#     results_old = ut.interpret_output(prob_label_old[0],w,h)
#     results = ut.interpret_output(prob_label[0],w,h)
#     
#     ut.show_results(src,results_old)
#     cv2.waitKey()
#     ut.show_results(src,results)












