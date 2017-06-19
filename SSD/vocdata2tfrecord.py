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

from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

###Code Reference http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

SIZE_PER_FILE = 128

data_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train'
tfrecords_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train_tfrecord'
v = van.vanilla_ssd_net('/gpu:1')


#### For test
#data_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/vocdemo/img'
#tfrecords_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/vocdemo/tfrecord'
#s = ssd_s.ssd_shrink_network('ssd_s08', 1,  batch_size, '', '/gpu:1')
#### For home
#data_path =  '/home/dashmoment/dataset/demo/img'
#tfrecords_path = '/home/dashmoment/dataset/demo/tfrecord/'
#v = van.vanilla_ssd_net('/gpu:0','/home/dashmoment/dataset/model/ssd_300/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt')



#img_path = os.path.join(data_path, filelist[0])
#img = cv2.imread(img_path)
#image_pro, fglabel, fglocation, fgscore = v.create_img_label(img)

filelist = os.listdir(data_path)


for i in range(len(filelist)//SIZE_PER_FILE):
    
    tfrecords_filename = os.path.join(tfrecords_path, 'voc_for_ssd_training_'+str(i)+'.tfrecords')
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    
    print("Process:{}/{}".format(i+1,len(filelist)//SIZE_PER_FILE))
    
    start = time.clock()
    for j in range(SIZE_PER_FILE):
        
      
        file_idx = i*SIZE_PER_FILE + j
        
        
        img_path = os.path.join(data_path, filelist[file_idx])
        img = cv2.imread(img_path)
        
        height = img.shape[0]
        width = img.shape[1]
       
        image_pro, fglabel, fglocation, fgscore = v.create_img_label(img)
         
    
        img_raw = img.tostring()
        img_process = image_pro.tostring()
        fglabel_b = fglabel.tostring()
        fglocation_b = fglocation.tostring()
        fgscore_b = fgscore.tostring()
        
        example = tf.train.Example(features=tf.train.Features(feature={
                
                'height':_int64_feature(height),
                'width':_int64_feature(width),
                'imgraw':_bytes_feature(img_raw),
                'imgpre':_bytes_feature(img_process),
                'fglabel':_bytes_feature(fglabel_b),
                'fglocation':_bytes_feature(fglocation_b),
                'fgscore':_bytes_feature(fgscore_b)
                }))
        
        writer.write(example.SerializeToString())
    writer.close()
    
    end = time.clock()
    print("time:{}".format(end-start))
    
    

















    
    
    
    
    
    