import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from nets import custom_layers

import utility as ut

class vanilla_ssd_net:
    
    def __init__(self):
        
        self.net_shape = (300, 300)
        self.data_format = 'NHWC'
        self.ckpt_filename = '/home/ubuntu/workspace/fastcnn/model/SSD_300/ssd_300_vgg.ckpt'
        

    def inference(self,img):
        
        net_shape = (300, 300)
        data_format = 'NHWC'
        ckpt_filename = '/home/ubuntu/workspace/fastcnn/model/SSD_300/ssd_300_vgg.ckpt'
        
        img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # Evaluation pre-processing: resize to SSD net shape.
        image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        image_4d = tf.expand_dims(image_pre, 0)
        
        reuse = True if 'ssd_net' in locals() else None
        ssd_net = ssd_vgg_300.SSDNet()
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            predictions, localisations,  logits, end_points = ssd_net.net(image_4d, is_training=False, reuse=reuse)
     
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess,ckpt_filename)
        
        
        # SSD default anchor boxes.
        ssd_anchors = ssd_net.anchors(net_shape)
        
        rlogit, rpredictions, rlocalisations, rbbox_img = self.sess.run([logits, predictions, localisations, bbox_img],
                                                                  feed_dict={img_input: img})
        
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                    rpredictions, rlocalisations, ssd_anchors,
                    select_threshold=0.5, img_shape=net_shape, num_classes=21, decode=True)
        
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=0.45)
        
        label = tf.placeholder(tf.int64, shape=(len(rclasses)))
        location = tf.placeholder(tf.float32, shape=(len(rclasses), 4))
        
        tlabel, tlocation, tscore = ssd_net.bboxes_encode(label, location, ssd_anchors)
        self.glabel, self.glocation, self.gscore = self.sess.run([tlabel, tlocation, tscore] , feed_dict={label:rclasses, location:rbboxes})
        
        return self.glabel, self.glocation, self.gscore
       
       







