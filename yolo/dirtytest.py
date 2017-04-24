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
import matplotlib.pyplot as plt


test_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_val"
img_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/JPEGImages'
labelfiles = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/ImageSets/Main'
graph_model = '../../model/yolo_dk/fcann_v1.ckpt-4000.meta'
checkpoint_dir = '../../model/yolo_dk'
classes = voc.list_image_sets()
val_list = voc.imgs_from_category_as_list('', 'val', labelfiles)

yolo_old = YOLO_tiny_tf.YOLO_TF()

#Vanilla YOLO_tiny Weight
modelTicket_G = {'root':'yolo_tiny', 'branch':'double_cut89'}

x = tf.placeholder(tf.float32,(None,448,448,3))
keep_prob = tf.placeholder(tf.float32)
gen_var = mut.create_var_xavier('train',mut.model_zoo(modelTicket_G))
yolo_ds = nf.yolo_dinception("yolo_train", x ,gen_var,keep_prob, False)


val_name = val_list[23]

resaver = tf.train.Saver()

tp = 0
fp = 0

tp_old = 0
fp_old = 0

num = 0
num_old = 0
idx = 1
elapse = 0
elapse_old = 0

with tf.Session() as sess2:
    
    
    resaver.restore(sess2, tf.train.latest_checkpoint(checkpoint_dir))
    c2 = sess2.run(gen_var["conv2w"])
    
    fpath = os.path.join('/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train','2008_000217.jpg')
    w,h,inputs = ut.vocimg_preprocess(fpath)
    src = cv2.imread(fpath)
    prob_label_old = yolo_old.sess.run(yolo_old.fc_19, feed_dict={yolo_old.x:inputs})
    prob_label = sess2.run(yolo_ds,feed_dict={x:inputs, keep_prob:0.5})
    
    results = ut.interpret_output(prob_label[0],w,h)
    results_old = ut.interpret_output(prob_label_old[0],w,h)
    
    ut.show_results(src,results_old)
    

    
    
    
    
    
    
    
    
    
    
    
        
        
        
        