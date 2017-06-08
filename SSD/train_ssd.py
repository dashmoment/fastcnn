import os
from random import shuffle

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from nets import custom_layers

import pickle
import ssd_shrink_network as ssd_s


img_shape=(300, 300)


def loadfrompickle(filepath):
    with open(filepath, 'rb') as f:
        
        obj = pickle.load(f)
        
    return obj

def random_batch(s_obj, img_path, label_path ,index, batch_size, shuffle_list = []):
    
    if shuffle_list == []:
        shuffle_list = os.listdir(img_path)
        shuffle(shuffle_list)
        
    batche_files = shuffle_list[index:index+batch_size]
    
    batch = []                                                                    
    logit_batch = []
    location_batch = []
    score_batch = []
    
    for i in range(len(batche_files)):
        tmp_path = os.path.join(img_path, batche_files[i])
        label_batch = os.path.join(label_path, batche_files[i] + '.pickle')
        
        label = loadfrompickle(label_batch)
        
        img = cv2.imread(tmp_path)
        img = s_obj.img_preprocessing(img)

        batch.append(img)
        logit_batch.append(label[0])
        location_batch.append(label[1])
        score_batch.append(label[2])
        
    batch  = np.vstack(batch)
    logit_batch = np.reshape(np.stack(logit_batch), [-1])
    location_batch = np.reshape(np.stack(location_batch),[-1, 4])
    score_batch = np.reshape(np.stack(score_batch), [-1])
    
    
    return shuffle_list, [batch,logit_batch, location_batch,  score_batch]
        




img_path = '/home/dashmoment/dataset/demo/img'
label_path = '/home/dashmoment/dataset/demo/label'
checkpoint_dir = ''

model_name = "ssd_s08"
ratio = 0.8

#batch = random_batch(img_path, 0, 5)
index = 0
batch_size = 6

init_epoch = 0 
total_epoch = 10
total_img = len(os.listdir(img_path))


ssd = ssd_s.ssd_shrink_network(model_name, ratio, batch_size, checkpoint_dir,'/gpu:0')



for epoch in range(init_epoch, total_epoch):
    
    index = 0
    shuffle_list =[]
    
    print("epoch {}".format(epoch))
    
    for i in range(total_img//batch_size):
        
        index = batch_size*i
        shuffle_list, batch = random_batch(ssd, img_path, label_path, index, batch_size, shuffle_list)

        _, loss = ssd.sess.run([ssd.solver, ssd.loss],  feed_dict={ssd.inputs: batch[0] , ssd.glabel:batch[1], ssd.glocation:batch[2], ssd.gscore:batch[3]})
        print("loss:{}".format(loss))


#with tf.Session() as sess:
    

















