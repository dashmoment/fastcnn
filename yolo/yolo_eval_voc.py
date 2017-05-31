import tensorflow as tf
import numpy as np
import yolo_netfactory as nf
import random_batch as rb
import YOLO_tiny_tf
import cv2
import voc_utils as voc
import os
import utility as ut
from bs4 import BeautifulSoup as soup
import model_utility as mut
import time
import model_utility as mu
import matplotlib.pyplot as plt

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
                        init_layers[i][1][-2] = 40131
                    if idx == 2:
                        init_layers[i][1][-2] = 32095
                   
          
                tf.get_variable(init_layers[i][0],init_layers[i][1], initializer=tf.contrib.layers.xavier_initializer())

                name_dict.append(init_layers[i][0])
                
        scope_dict[scope_name] = name_dict 

    return scope_dict



img_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2007/Test/JPEGImages'
labelfiles = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2007/Test/ImageSets/Main'
checkpoint_dir = '../../model/yololoss_init_0.8'
classes = voc.list_image_sets()
val_list = voc.imgs_from_category_as_list('', 'test', labelfiles)

yolo_old = YOLO_tiny_tf.YOLO_TF()
varscope = 'recursive_1'
shrinkratio = 0.8

with tf.device('/gpu:0'):
#Vanilla YOLO_tiny Weight
    x = tf.placeholder(tf.float32,(None,448,448,3))
    label = tf.placeholder(tf.float32,(None,1470), name='labels')
    keep_prob = tf.placeholder(tf.float32)
    
    modelTicket_G = {'root':'yolo_tiny', 'branch':'vanilla'}
    init_layers = mu.model_zoo(modelTicket_G)
    var_dict = recursive_create_var('recursive', 2, shrinkratio, init_layers)
    yolo_ds = nf.glosso_train(varscope, 'test', x, var_dict, keep_prob, False)


    tlossTicket = {'loss':'smoothL1'}
    loss_pair = {'prob':yolo_ds}
    loss = mu.loss_zoo(tlossTicket, loss_pair, label)


tp = 0
fp = 0

tp_old = 0
fp_old = 0

num = 0
idx = 1
elapse = 0
elapse_old = 0
iou_threshold = 0.2


config = tf.ConfigProto()
config.gpu_options.allow_growth=True

val_name = val_list[500]

with tf.Session(config = config) as sess:
    
    resaver = tf.train.Saver()
    resaver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
#    c2 = sess2.run(ds_yolo["conv2w"])
    
    for val_name in val_list:
#    if val_name != []:
        
        fpath = os.path.join(img_root,val_name+'.jpg')
        w,h,inputs = ut.vocimg_preprocess(fpath)
        src = cv2.imread(fpath)
          
        num = num + ut.calc_objec_num(val_name)
        print("Test File:{}/{}".format(idx,len(val_list)))
        idx = idx + 1
        
        
       
        s = time.clock()
        prob_label_old = yolo_old.sess.run(yolo_old.fc_19, feed_dict={yolo_old.x:inputs})
        e = time.clock()
        elapse_old = elapse_old + e - s
        
        results_old = ut.interpret_output(prob_label_old[0],w,h)

        for i in range(len(results_old)):
            
            tbb_old = ut.cov_yoloBB2VOC(results_old[i])
            res_old = ut.eval_by_obj(val_name, tbb_old, iou_threshold)
            
            if(res_old == 1): tp_old = tp_old +1
            if(res_old == -1): fp_old = fp_old +1
        

        
        s = time.clock()
        prob_label = sess.run(yolo_ds,feed_dict={x:inputs, keep_prob:1})
        e = time.clock()
        elapse = elapse + e-s

        
        
        results = ut.interpret_output(prob_label[0],w,h)
        for i in range(len(results)):
            
            tbb = ut.cov_yoloBB2VOC(results[i])
            res = ut.eval_by_obj(val_name, tbb, iou_threshold)
            
            if(res == 1): tp = tp +1
            if(res == -1): fp = fp +1
        
        feeddict={x: inputs , label:prob_label_old}
        print(sess.run(loss,feed_dict=feeddict))
       
    print("Old Avg Elapse:{}".format(elapse_old/idx)) 
    print("Old Accuracy:{}".format(tp_old/num))
    
    print("New Avg Elapse:{}".format(elapse/idx)) 
    print("New Accuracy:{}".format(tp/num)) 
        
    
    
    ut.show_results(src,results_old)
    cv2.waitKey()
    ut.show_results(src,results)
    

    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        