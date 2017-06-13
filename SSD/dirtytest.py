import os
import matplotlib.image as mpimg
import vanilla_ssd as van
import pickle
import cv2
import tensorflow as tf

from datasets import dataset_factory
from preprocessing import preprocessing_factory
import ssd_shrink_network as ssd_s
import numpy as np
from random import shuffle
import time

data_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/vocdemo/img'
output_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/vocdemo/label'
logfile = '/home/ubuntu/workspace/fastcnn/log/test'


batch_size = 16
s = ssd_s.ssd_shrink_network('ssd_s08', 0.8,  batch_size, '', '/gpu:1')

image_list = os.listdir(data_path)

fcount = 0

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
    logit_batch = np.concatenate(logit_batch, axis=0)
    location_batch = np.concatenate(location_batch, axis=0)
    score_batch = np.concatenate(score_batch, axis=0)
    
    return shuffle_list, batch, logit_batch, location_batch, score_batch


epoch = 10

with tf.variable_scope('ssd_s08') as scope:
            scope.reuse_variables()
            var_s = tf.get_variable('conv1/conv1_1/weights')
v = s.sess.run(var_s)

for fid in range(1):
    
    fcount = fcount + 1
    
    
    
    for i in range(epoch):
        
        shuffle_list = []
        
        for j in range(len(image_list)//batch_size):
            
            print("epoch:{}, interation:{}/{}".format(i, j, len(image_list)//batch_size))
            
            index = batch_size*j
            
            st = time.clock()
    
            shuffle_list,  batch, logit_batch, location_batch, score_batch = random_batch(s ,data_path , output_path, index , batch_size, shuffle_list)
            
            e = time.clock()
            
            print("Radom Batch Elapse:{}".format(e-st))
            
            st = time.clock()
            loss, _ = s.sess.run([s.loss, s.solver], feed_dict={s.inputs:batch, s.glabel:logit_batch,  s.glocation:location_batch , s.gscore:score_batch})
            e = time.clock()
            print("Training Elapse:{}".format(e-st))
            print(loss)



















