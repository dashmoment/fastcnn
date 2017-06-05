import tensorflow as tf
import numpy as np
import yolo_netfactory as nf
import random_batch as rb
import cv2
import voc_utils as voc
import os
import utility as ut
import model_utility as mut
import time
import model_utility as mu
import matplotlib.pyplot as plt
import YOLO_tiny_tf


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
checkpoint_dir = '../../model/l1norm_entropy_init0.8'
classes = voc.list_image_sets()
val_list = voc.imgs_from_category_as_list('', 'test', labelfiles)

varscope = 'recursive_1'
shrinkratio = 0.8

#yolo_old = YOLO_tiny_tf.YOLO_TF()

with tf.device('/gpu:0'):

    x = tf.placeholder(tf.float32,(None,448,448,3))
    label = tf.placeholder(tf.float32,(None,1470), name='labels')
    keep_prob = tf.placeholder(tf.float32)
    
    modelTicket_G = {'root':'yolo_tiny', 'branch':'vanilla'}
    init_layers = mu.model_zoo(modelTicket_G)
    var_dict = recursive_create_var('recursive', 2, shrinkratio, init_layers)
    yolo_ds = nf.glosso_train(varscope, 'test', x, var_dict, keep_prob, False)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config = config) as sess:
    
    resaver = tf.train.Saver()
    resaver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    
    
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        h_img, w_img,_ = frame.shape
        img_resized = cv2.resize(frame, (448, 448))
        img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray( img_RGB )
        
        res = np.zeros((1,448,448,3),dtype='float32')
        res[0] = (img_resized_np/255.0)*2.0-1.0

        s = time.clock()
        prob_label = sess.run(yolo_ds,feed_dict={x:res, keep_prob:1})
        #prob_label = yolo_old.sess.run(yolo_old.fc_19, feed_dict={yolo_old.x:res})
        e = time.clock()
        elapse = e-s

        results = ut.interpret_output(prob_label[0],w_img,h_img)

        print(elapse)
       
        ut.show_results(frame, results, 1/elapse)

       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break






















