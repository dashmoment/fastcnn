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

###Code Reference http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

data_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/vocdemo/img'

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#tfrecords_filename = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/vocdemo/tfrecord/pascal_voc_segmentation.tfrecords'
#writer = tf.python_io.TFRecordWriter(tfrecords_filename)
#
#filelist = os.listdir(data_path)
#for i in range(len(filelist)):
#
#
#    img_path = os.path.join(data_path, filelist[0])
#    img = np.array(Image.open(img_path))
#    
#    height = img.shape[0]
#    width = img.shape[1]
#    
#    img_raw = img.tostring()
#    
#    example = tf.train.Example(features=tf.train.Features(feature={
#            
#            'height':_int64_feature(height),
#            'width':_int64_feature(width),
#            'imgraw':_bytes_feature(img_raw)
#            }))
#    
#    writer.write(example.SerializeToString())
#writer.close()


def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'imgraw': tf.FixedLenFeature([], tf.string),
        })
    
    image = tf.decode_raw(features['imgraw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)
    
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                           target_height=300,
                                           target_width=300)
    
#    height = tf.cast(features['height'], tf.int32)
#    width = tf.cast(features['width'], tf.int32)
    
    images = tf.train.shuffle_batch( [resized_image],
                                         batch_size=32,
                                         capacity=60,
                                         num_threads=10,
                                         min_after_dequeue=10)
    
    return images
    
filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=10)   

img = read_and_decode(filename_queue) 
    
    
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session()  as sess:
    
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)  
    
    for i in range(100):
    
        img_s = sess.run(img)
        print("{}/{}".format(i,100))
    
    coord.request_stop()
    coord.join(threads)
    
    
    
    
    
    
    
    
    
    
    
    