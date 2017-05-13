import tensorflow as tf
import numpy as np
from random import shuffle
import os
from scipy import misc


def image_random_batch(dirname, batchsize, imagesize, array_type):
    
    cwd = os.getcwd()
    os.chdir(dirname)
    
    files = [x for x in os.listdir(dirname)]
    idx = [x for x in range(len(files))]
    shuffle(idx)
    
    batch = [files[b] for b in idx[0:batchsize]]
    batch_images = []

    for fname in batch:
        tmp = misc.imread(fname).astype(array_type)
        tmp = misc.imresize(tmp,imagesize).astype(array_type)
        batch_images.append(np.array(tmp))
    
    batch_images = np.stack(batch_images)
    
    os.chdir(cwd)
    
    return batch_images

dname = "/home/dashmoment/workspace/dataset/VOCdevkit/VOC2012/JPEGImages"
b = image_random_batch(dname, 128, (255,255,3), np.float32)


x = tf.placeholder(tf.float32,(None,255,255,3))
image_submean = tf.subtract(x, tf.reduce_mean(x))

rconv1_w = tf.Variable(tf.random_normal([11,11,3,48]))
rconv1_b = tf.Variable(tf.random_normal([48]))
rconv1_in = tf.nn.conv2d(image_submean, rconv1_w, strides=[1,2,2,1], padding='SAME')
rconv_in = tf.nn.bias_add(rconv1_in, rconv1_b)
rconv1_in = tf.nn.relu(rconv_in)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)  
    
    p = sess.run(rconv1_in, feed_dict={x:b})