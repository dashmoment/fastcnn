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


batch_size = 16
SIZE_PER_FILE = 128

tfrecords_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train_tfrecord'
model_path = '/home/ubuntu/workspace/fastcnn/model/ssd_v1_08'
model_file = '/home/ubuntu/workspace/fastcnn/model/ssd_v1_08/ssd_v1_08.ckpt'
log_path = '/home/ubuntu/workspace/fastcnn/log/ssd_v1_08'


if len(os.listdir(model_path)) > 0:
    s = ssd_s.ssd_shrink_network('ssd_s08', 0.8,  batch_size, model_path, '/gpu:1')
else:
    s = ssd_s.ssd_shrink_network('ssd_s08', 0.8,  batch_size, '', '/gpu:1')


def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()#
    _, serialized_example = reader.read(filename_queue)#
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'imgraw': tf.FixedLenFeature([], tf.string),
        'imgpre':tf.FixedLenFeature([], tf.string),
        'fglabel': tf.FixedLenFeature([], tf.string),
        'fglocation': tf.FixedLenFeature([], tf.string),
        'fgscore': tf.FixedLenFeature([], tf.string)
        })
    
    image = tf.decode_raw(features['imgpre'], tf.float32)
    image_raw = tf.decode_raw(features['imgraw'], tf.uint8)
    
    
    tfglabel = tf.decode_raw(features['fglabel'], tf.int64)
    tfglocation = tf.decode_raw(features['fglocation'], tf.float32)
    tfgscore = tf.decode_raw(features['fgscore'], tf.float32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image_shape = tf.stack([height, width, 3])
    
    image = tf.reshape(image, (300,300,3))
    image_raw = tf.reshape(image_raw, image_shape)
    tfglabel = tf.reshape(tfglabel, [8732])
    tfglocation = tf.reshape(tfglocation, [8732,4])
    tfgscore = tf.reshape(tfgscore, [8732])   
    
    
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image_raw,
                                                           target_height=300,
                                                           target_width=300)
    
    image_r, images, tfglabel, tfglocation, tfgscore, theight, twidth = tf.train.shuffle_batch( [resized_image, image ,tfglabel, tfglocation, tfgscore, height, width],
                                         batch_size=batch_size,
                                         capacity=600,
                                         num_threads=3,
                                         min_after_dequeue=0)
    
    
    
    
    return image_r, images, tfglabel, tfglocation, tfgscore, [theight, twidth]
    

image_raw, img, label, loc, score, img_size  = read_and_decode(s.train_filename_queue) 
timage_raw, timg, tlabel, tloc, tscore, timg_size  = read_and_decode(s.test_filename_queue) 


coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=s.sess, coord=coord)  

tf.summary.scalar("Train Loss",s.loss, collections=['train'])
tf.summary.scalar("Test_Loss",s.loss, collections=['test'])
merged_summary_train = tf.summary.merge_all('train')
merged_summary_test = tf.summary.merge_all('test')

summary_writer = tf.summary.FileWriter(log_path, s.sess.graph) 
saver = tf.train.Saver()

try:
    iteration= 0 
    while not coord.should_stop():
        
        
        iteration = iteration + 1
        print("Iteration:{}".format(iteration))

        image_r, img_s, tfglabel_s, loc_s, score_s, sizes = s.sess.run([image_raw, img, label, loc, score, img_size])
        
        tfglabel_s = np.reshape(tfglabel_s, [-1])
        loc_s = np.reshape(loc_s, [-1,4])
        score_s = np.reshape(score_s, [-1])
        
        s.sess.run(s.solver, feed_dict={s.inputs:img_s, s.glabel:tfglabel_s,  s.glocation:loc_s , s.gscore:score_s})
        

        if iteration%5 == 0:

            loss, train_sum = s.sess.run([s.loss, merged_summary_train], feed_dict={s.inputs:img_s, s.glabel:tfglabel_s,  s.glocation:loc_s , s.gscore:score_s})
            summary_writer.add_summary(train_sum, iteration)
            saver.save(s.sess, model_file, global_step=iteration)
            print("loss:{}".format(loss))

        if iteration%6 == 0:

            timg_s, ttfglabel_s, tloc_s, tscore_s = s.sess.run([timg, tlabel, tloc, tscore])
            tloss, test_sum = s.sess.run([s.loss, merged_summary_test], feed_dict={s.inputs:timg_s, s.glabel:ttfglabel_s,  s.glocation:tloc_s , s.gscore:tscore_s})
            summary_writer.add_summary(train_sum, test_sum)
            print("Test loss:{}".format(tloss))
           
                   
        
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()

coord.request_stop()
coord.join(threads)





















    
    
    
    
    
    