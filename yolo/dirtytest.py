#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:59:54 2017

@author: ubuntu
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import yolo_netfactory as nf
import random_batch as rb
import YOLO_tiny_tf
import cv2

import time

#root_dir = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012'
#img_dir = os.path.join(root_dir, 'JPEGImages/')
#dst_dir = os.path.join(root_dir, 'voc_train/')
#tdst_dir = os.path.join(root_dir, 'voc_test/')
#
#cat_all = voc_utils.list_image_sets()
#
#
#
#with open('train.pickle', 'wb') as handle:
#
#	path = []
#
#	for cat in cat_all:
#		
#		imlist = voc_utils.imgs_from_category_as_list("train",cat)
#		timlist = voc_utils.imgs_from_category_as_list("val",cat)
#
#		for i in imlist:
#			imgpath = os.path.join(img_dir, i+".jpg")
#			dstpath = os.path.join(dst_dir, i+".jpg")
#
#			path.append(imgpath)
#			
#			
#			#copyfile(imgpath, dstpath)
#			#print(imgpath)
#
#		for j in timlist:
#			imgpath = os.path.join(img_dir, j+".jpg")
#			dstpath = os.path.join(tdst_dir, j+".jpg")
#			path.append(imgpath)
#			#pickle.dump(imgpath, handle, protocol=pickle.HIGHEST_PROTOCOL)
#			#copyfile(imgpath, dstpath)
#			#print(imgpath)
#	pickle.dump(path, handle, protocol=pickle.HIGHEST_PROTOCOL)
#	handle.close()
#
#with open('train.pickle', 'rb') as handle:
#	unserialized_data = pickle.load(handle)
#	
#
#	print(unserialized_data[0])
#	print (len(unserialized_data))
#handle.close()
#ann = voc_utils.load_annotation(iml[0])
#mask = voc_utils.get_masks("train",cat[0],"bbox1")
#fromfile = "test/2008_000090.jpg"
#
#yolo = YOLO_tiny_tf.YOLO_TF()
#
#img = cv2.imread(fromfile)
#
#img_resized = cv2.resize(img, (448, 448))
#img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
#img_resized_np = np.asarray( img_RGB )
#inputs = np.zeros((1,448,448,3),dtype='float32')
#inputs[0] = (img_resized_np/255.0)*2.0-1.0
#
#c = tf.Variable(tf.truncated_normal([3,3,3,16], mean=0, stddev=0.01))
#d = tf.Variable(tf.truncated_normal([16], mean=0, stddev=0.01))
#
#var_dict = {}
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    in_dict = {yolo.x: inputs}
#    net_output = yolo.sess.run(yolo.fc_19,feed_dict={yolo.x: inputs})
#    
#    print(net_output)
#    
#    tensor_to_load_1 = yolo.sess.run(tf.get_default_graph().get_tensor_by_name('Variable:0'))
#    var_dict['Variable'] = tensor_to_load_1
#    tensor_to_load_1 = yolo.sess.run(tf.get_default_graph().get_tensor_by_name('Variable_1:0'))
#    var_dict['Variable_1'] = tensor_to_load_1
#    
#    print(var_dict)
#    
##    yolo.detect_from_file(fromfile)
#    op = tf.assign(c, var_dict['Variable'])
#    op1 = tf.assign(d, var_dict['Variable_1'])
#    cr =  sess.run(op)
#    cr1 =  sess.run(op1)


def init_yolo_weight(sess,yolo_cls, ds_yolo):
    
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
    
    for var in key_pairs:
        yolo_var = yolo.sess.run(tf.get_default_graph().get_tensor_by_name(key_pairs[var]+':0'))
        op = tf.assign(ds_yolo[var], yolo_var)
        sess.run(op)
    



batch_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train"
test_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_val"
filename = "../../model/test/fcann_v1.ckpt"
logfile = '../../log/test'
graph_model = '../../model/test/fcann_v1.ckpt-65000.meta'
checkpoint_dir = '../../model/test'

#Vanilla Yolo
yolo = YOLO_tiny_tf.YOLO_TF()

ds_yolo = {
        'conv1w':tf.Variable(tf.truncated_normal([3,3,3,16], mean=0, stddev=0.01)),
        'conv1b':tf.Variable(tf.truncated_normal([16], mean=0, stddev=0.01)),
        'conv2w':tf.Variable(tf.truncated_normal([3,3,16,32], mean=0,stddev=0.01)),
        'conv2b':tf.Variable(tf.truncated_normal([32], mean=0, stddev=0.01)),
        'conv3w':tf.Variable(tf.truncated_normal([3,3,32,64], mean=0, stddev=0.01)),
        'conv3b':tf.Variable(tf.truncated_normal([64], mean=0, stddev=0.01)),
        'conv4w':tf.Variable(tf.truncated_normal([3,3,64,128], mean=0, stddev=0.01)),
        'conv4b':tf.Variable(tf.truncated_normal([128], mean=0, stddev=0.01)),
        'conv5w':tf.Variable(tf.truncated_normal([3,3,128,256], mean=0, stddev=0.01)),
        'conv5b':tf.Variable(tf.truncated_normal([256], mean=0, stddev=0.01)),
        'conv6w':tf.Variable(tf.truncated_normal([3,3,256,512], mean=0, stddev=0.01)),
        'conv6b':tf.Variable(tf.truncated_normal([512], mean=0, stddev=0.01)),
        'conv7w':tf.Variable(tf.truncated_normal([3,3,512,1024], mean=0, stddev=0.01)),
        'conv7b':tf.Variable(tf.truncated_normal([1024], mean=0, stddev=0.01)),
        'conv8w':tf.Variable(tf.truncated_normal([3,3,1024,1024], mean=0, stddev=0.01)),
        'conv8b':tf.Variable(tf.truncated_normal([1024], mean=0, stddev=0.01)),
        'conv9w':tf.Variable(tf.truncated_normal([3,3,1024,1024], mean=0, stddev=0.01)),
        'conv9b':tf.Variable(tf.truncated_normal([1024], mean=0, stddev=0.01)),
        'fc10w':tf.Variable(tf.truncated_normal([50176,256], mean=0, stddev=0.01)),
        'fc10b':tf.Variable(tf.truncated_normal([256], mean=0, stddev=0.01)),
        'fc11w':tf.Variable(tf.truncated_normal([256,4096], mean=0, stddev=0.01)),
        'fc11b':tf.Variable(tf.truncated_normal([4096], mean=0, stddev=0.01)),
        'fc12w':tf.Variable(tf.truncated_normal([4096,1470], mean=0, stddev=0.01)),
        'fc12b':tf.Variable(tf.truncated_normal([1470], mean=0, stddev=0.01))
        }

x = tf.placeholder(tf.float32,(None,448,448,3))
keep_prob = tf.placeholder(tf.float32)

fromfile = "test/person.jpg"
img = cv2.imread(fromfile)
img_resized = cv2.resize(img, (448, 448))
img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
img_resized_np = np.asarray( img_RGB )
inputs = np.zeros((1,448,448,3),dtype='float32')
inputs[0] = (img_resized_np/255.0)*2.0-1.0

#inputs = rb.yolo_image_random_batch(batch_file, 64, (448,448,3), np.float32)

yolo_ds = nf.yolo_vanilla(x,ds_yolo,keep_prob)
yolo.h_img = 448
yolo.w_img = 448

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())  
    init_yolo_weight(sess,yolo,ds_yolo)
    c1 = sess.run(ds_yolo['conv1w'])
    c2 = sess.run(ds_yolo['fc10w'])
    
    res_n = sess.run(yolo_ds, feed_dict={x:inputs, keep_prob:0.5})
    res = yolo.sess.run(yolo.fc_19, feed_dict={yolo.x:inputs})
    res_value = tf.subtract(res, res_n)
    loss = tf.sqrt(tf.reduce_sum(tf.square(res_value)))
    
    cost = sess.run(loss)
    print("Total Loss:{}".format(cost))
    
    results = yolo.interpret_output(res[0])
    yolo.show_results(img_resized,results)
    
    results = yolo.interpret_output(res_n[0])
    yolo.show_results(img_resized,results)
    
    
#    filter_mat_probs = np.array(probs>=0.2,dtype='bool')
#    filter_mat_boxes = np.nonzero(filter_mat_probs)
#    boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
#    probs_filtered = probs[filter_mat_probs]
#    classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 
#
#    argsort = np.array(np.argsort(probs_filtered))[::-1]
#    boxes_filtered = boxes_filtered[argsort]
#    probs_filtered = probs_filtered[argsort]
#    classes_num_filtered = classes_num_filtered[argsort]
#    
#    for i in range(len(boxes_filtered)):
#        if probs_filtered[i] == 0 : continue
#        for j in range(i+1,len(boxes_filtered)):
#            if yolo.iou(boxes_filtered[i],boxes_filtered[j]) > yolo.iou_threshold : 
#                probs_filtered[j] = 0.0
#    
#    filter_iou = np.array(probs_filtered>0.0,dtype='bool')
#    boxes_filtered = boxes_filtered[filter_iou]
#    probs_filtered = probs_filtered[filter_iou]
#    classes_num_filtered = classes_num_filtered[filter_iou]
#
#    result = []
#    for i in range(len(boxes_filtered)):
#        result.append([yolo.classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

    
    # def detect_from_cvmat(img):
#        s = time.time()
#        h_img,w_img,_ = img.shape
#        img_resized = cv2.resize(img, (448, 448))
#        img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
#        img_resized_np = np.asarray( img_RGB )
#        inputs = np.zeros((1,448,448,3),dtype='float32')
#        inputs[0] = (img_resized_np/255.0)*2.0-1.0
#        in_dict = {self.x: inputs}
#        net_output = self.sess.run(self.fc_19,feed_dict=in_dict)
#        self.result = self.interpret_output(net_output[0])
#        self.show_results(img,self.result)
#                      
#        strtime = str(time.time()-s)
#        if self.disp_console : print ('Elapsed time : ' + strtime + ' secs' + '\n')
#for var in ds_yolo:
#    tf.summary.histogram(var, ds_yolo[var], collections=['train'])
#
#     
#continue_training = 1
#loop_num = 0
#batch_size = 0
#
#keep_prob = tf.placeholder(tf.float32)
#x = tf.placeholder(tf.float32,(None,448,448,3))
#label = tf.placeholder(tf.float32,(None,1470))
#
##Train Phase
#yolo_ds_train = nf.yolo_vanilla_train(x,ds_yolo,keep_prob)
##yolo_ds_train = nf.yolo_ds_train(x,ds_yolo,keep_prob)
#res_value = tf.subtract(yolo_ds_train, label)
#loss = tf.sqrt(tf.reduce_sum(tf.square(res_value)))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
#tf.summary.scalar("train_RMSE",loss, collections=['train'])
#
##Test Phase
#yolo_ds = nf.yolo_vanilla(x,ds_yolo,keep_prob)
##yolo_ds = nf.yolo_ds(x,ds_yolo,keep_prob)
#tres_value = tf.subtract(yolo_ds, label)
#tloss = tf.sqrt(tf.reduce_sum(tf.square(tres_value)))
#tf.summary.scalar("test_RMSE",tloss, collections=['test'])
#
#merged_summary_train = tf.summary.merge_all('train')
#merged_summary_test= tf.summary.merge_all('test')
#
#with tf.Session() as sess2:
#    
#    summary_writer = tf.summary.FileWriter(logfile, sess2.graph)  
#    saver = tf.train.Saver()    
#    sess2.run(tf.global_variables_initializer())  
#    
#    if continue_training !=0:
#        
#        resaver = tf.train.import_meta_graph(graph_model)
#        resaver.restore(sess2, tf.train.latest_checkpoint(checkpoint_dir))
#        
#        for var in ds_yolo:
#            sess2.run(ds_yolo[var].assign(tf.get_collection(var)[0]))
#        
#        continue_training = 0
#    
#    for i in range(loop_num, 200000000):
#        
#        print("Epoch:{}".format(i))
#        image_src =  rb.yolo_image_random_batch(batch_file, batch_size, (448,448,3), np.float32)
#        prob_label = yolo.sess.run(yolo.fc_19, feed_dict={yolo.x:image_src})
#        sess2.run(train_step, feed_dict={x:image_src, label:prob_label, keep_prob:0.5})
#        
#               
#        if i%500 == 0:
#         
#            train_loss, sumtrain = sess2.run([loss,merged_summary_train], feed_dict={x:image_src, label:prob_label, keep_prob:0.5})         
#            print("Train Loss: {}".format(train_loss))
#            summary_writer.add_summary(sumtrain, i)
#            
#            for var in ds_yolo:          
#                tf.add_to_collection(var, ds_yolo[var])
#            saved_model = saver.save(sess2, filename, global_step=i)
#            
#            if i%1000 == 0:
#                image_test = rb.yolo_image_random_batch(test_file, batch_size, (448,448,3), np.float32)
#                prob_label2 = yolo.sess.run(yolo.fc_19, feed_dict={yolo.x:image_test})
#                test_loss, sumtest = sess2.run([tloss,merged_summary_test], feed_dict={x:image_test, label:prob_label2, keep_prob:0.5})
#                print("Test Loss: {}".format(test_loss))
#                summary_writer.add_summary(sumtest, i)
#    
#    
#    
#    summary_writer.close()
#    sess2.close()  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




