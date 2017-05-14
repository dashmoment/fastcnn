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
                    
                    shape[-1] = int(init_layers[i][1][-1]*reduce_percent)
                    
                    
                init_layers[i][1] = shape
                
                if init_layers[i][0] == 'fc12w':  init_layers[i][1][-1] = 1470
                if init_layers[i][0] == 'fc12b':  init_layers[i][1][-1] = 1470          
                if init_layers[i][0] == 'fc10w':
                
                    if idx == 0:
                        init_layers[i][1][-2] = 50176
                    if idx == 1:
                        init_layers[i][1][-2] = 47628
                    if idx == 2:
                        init_layers[i][1][-2] = 32095
                   
          
                tf.get_variable(init_layers[i][0],init_layers[i][1], initializer=tf.contrib.layers.xavier_initializer())

                name_dict.append(init_layers[i][0])
                
        scope_dict[scope_name] = name_dict 

    return scope_dict


img_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2007/Test/JPEGImages'
shrink_ratio = 0.95

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
    var_dict = recursive_create_var('recursive', 3, shrink_ratio, init_layers)
    
x = tf.placeholder(tf.float32,(None,448,448,3))
label = tf.placeholder(tf.float32,(None,1470), name='labels')
keep_prob = tf.placeholder(tf.float32)

var_scope = 'recursive_1'

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config = config) as sess:
    
     sess.run(tf.global_variables_initializer())  
     
     
#     mu.weight_pruning_ind(yolo_old, sess, var_dict[var_scope], var_scope)
     
     
     mu.weight_pruning(yolo_old, sess, var_dict[var_scope], var_scope)


     fpath = '/home/dashmoment/workspace/dataset/VOCdevkit/VOC2007/Test/JPEGImages/000002.jpg'    
     w,h,inputs = ut.vocimg_preprocess(fpath)
     src = cv2.imread(fpath)
     
     yolo_ds = nf.glosso_train(var_scope, 'test', x, var_dict, keep_prob, False)  
     
     for i in range(10):
    
         
         
         s = time.clock()
         prob_label_old = yolo_old.sess.run(yolo_old.fc_19, feed_dict={yolo_old.x:inputs})
         e = time.clock()    
         print("Time old:{}".format(e-s))
         
         s = time.clock()     
         prob_label = sess.run(yolo_ds,feed_dict={x:inputs, keep_prob:1})
         e = time.clock()
         print("Time:{}".format(e-s))
     
     
     label = prob_label_old
     predict = prob_label
     label_cls = tf.slice(label, [0,0],[-1, 980])
     label_offset = tf.slice(label, [0, 980],[-1, 490])

     pre_cls = tf.slice(predict, [0,0],[-1, 980])
     pre_offset = tf.slice(predict, [0, 980],[-1, 490])

    #===========KL Divergence of each class====================
     label_cls = tf.nn.softmax(tf.reshape(label_cls, [-1,49,20] ))
     pre_cls = tf.nn.softmax(tf.reshape(pre_cls, [-1,49,20]))
     kl_loss = tf.reduce_sum(tf.multiply(pre_cls, tf.log(tf.divide(pre_cls,label_cls))), axis=[1,2])

    #============SmoothL1 for rest parts======================
     res_value = tf.abs(tf.subtract(pre_offset,label_offset))
     smoothL1 = tf.cast(tf.less(res_value,1), tf.float32)
     invsmoothL1 = tf.cast(tf.less(smoothL1,0.5),tf.float32)
     r1 = tf.multiply(tf.square(res_value), smoothL1)*0.5
     r2 = tf.multiply((res_value - 0.5), invsmoothL1)       
     offset_loss = tf.reduce_sum(tf.add(r1,r2), axis=1)#
     
     skl_loss = sess.run(kl_loss)
     soffset_loss = sess.run(offset_loss)
     loss = sess.run(tf.add(kl_loss, offset_loss))
     
     l_cls = sess.run(label_cls)
     p_cls = sess.run(pre_cls)

#     results_old = ut.interpret_output(prob_label_old[0],w,h)
#     results = ut.interpret_output(prob_label[0],w,h)
#     
#     ut.show_results(src,results_old)
#     cv2.waitKey()
#     ut.show_results(src,results)
     
     tp = 0
     
     for k in range(49):
         if np.argmax(l_cls[0,k,:]) == np.argmax(p_cls[0,k,:]):
             tp =tp + 1
         












