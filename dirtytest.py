import net_factory
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
from scipy.misc import imread
from scipy.misc import imresize
from caffe_classes import class_names
from PIL import Image
import matplotlib.pyplot as plt

def randombatch():
    filenames = tf.train.match_filenames_once("../../dataset/ilsvrc_train/*.jpg")
    file_queue =  tf.train.string_input_producer( filenames)
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(file_queue)
    image_orig = tf.image.decode_jpeg(image_file)
    image = tf.image.resize_images(image_orig, [227, 227])
    image.set_shape((227, 227, 3))   
    num_preprocess_threads = 5
    min_queue_examples = 500
    batch_size = 500
    
    images = tf.train.shuffle_batch(
            [image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + (num_preprocess_threads +50)*batch_size,
            min_after_dequeue=min_queue_examples)
    
    return images

y = np.array([1,0,1,0,0])

data = np.array([[1,0,1,0,0],[10,0,1,10,0],[1,5,1,0,0],[1,0,1,0,3],[1,3,1,0,0]])

x = tf.placeholder(tf.float32,(None,5))
w1 = tf.Variable(tf.random_normal([5,3]))

w1_2 = tf.concat([w1,w1],1)
w2 = tf.Variable(tf.random_normal([6,1]))

b1 = tf.Variable(tf.random_normal([6]))
b2 = tf.Variable(tf.random_normal([1]))

fc1 = tf.nn.relu_layer(x, w1_2, b1)
fc2 = tf.nn.relu_layer(fc1, w2, b2)

loss = tf.subtract(fc2, y)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
       
    for i in range(1000):
        
        sess.run(train_step, feed_dict={x:data})
        print("w_one:{}".format(sess.run(w1)))
        print("w_double:{}".format(sess.run(w1_2)))
        print("loss:{}".format(sess.run(fc2, feed_dict={x:data})))


#
#sample_batch = randombatch()
 
#with tf.Session() as sess:
    
#    new_saver = tf.train.import_meta_graph('../model/full_dim/fcann_v1.ckpt-8.meta')
    #new_saver.restore(sess, tf.train.latest_checkpoint('../model'))
#    new_saver.restore(sess, '../model/full_dim/fcann_v1.ckpt-8')
    
    
    
#    tw1 = tf.Variable(sess.run(tf.get_collection("w1")[0]))
#    tb1 = tf.Variable(sess.run(tf.get_collection("b1")[0]))
#    sess.run(tf.global_variables_initializer())
    
#    tw2 = sess.run(tw1)
    
#    exp = tf.concat([tw1,tw1],3)
   
    
#   res = sess.run(exp)
    
    
    
#    
#    net = net_factory.vanilla_alex_full(x)
#    mininet = net_factory.mini_alex_full(x,tw1,tb1)
#    
#    sess.run(tf.global_variables_initializer())
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord) 
#    
#    try:
#        
#        image_tensor = sess.run(sample_batch)
#        w1 = sess.run(tw1)
#        output = sess.run(net, feed_dict = {x:image_tensor})
#        output2 = sess.run(mininet, feed_dict = {x:image_tensor})
#        
#        plt.figure(1)
#        plt.subplot(211)
#        plt.plot(output[0])
#        plt.plot(output2[0])
#        
#        for input_im_ind in range(output.shape[0]):
#            
#            inds = np.argsort(output)[input_im_ind,:]
#            inds2 = np.argsort(output2)[input_im_ind,:]
#            
#                
##            for i in range(5):
##                print("Image", input_im_ind)
##                print("Origin:")
##                print(class_names[inds[-1-i]], output[input_im_ind, inds[-1-i]])
##                print("Small:")
##                print(class_names[inds[-1-i]], output2[input_im_ind, inds[-1-i]])
#                
#            for j in range(0,len(inds)-5):
#                output[input_im_ind, inds[j]] = 0
#           
#        plt.subplot(212)
#        plt.plot(output[0])
#        plt.plot(output2[0])
#        cost = tf.reduce_mean(-tf.reduce_sum(tf.multiply(output,tf.log(output2)),1))
#        loss = sess.run(cost)
#        
#        
#        print("Avg Loss:{}".format(loss))
#    except tf.errors.OutOfRangeError:
#        print('Done training -- epoch limit reached')
#    finally:
#        # When done, ask the threads to stop.
#        coord.request_stop()
#        coord.join(threads)
#    sess.close()
    
    
    
###!/usr/bin/env python3
### -*- coding: utf-8 -*-
##"""
##Created on Thu Mar 16 12:40:11 2017
##
##@author: ubuntu
##"""
##
##import purgeinvalid_img as pi
##import numpy as np
##
###pi.purgeinvalid_img("./ilsvrc11")
###
###pi.test();
##
##a = np.array([[[3, 3, 3],[4, 4, 4]],[[1, 3, 3],[2, 4, 4]]])
##b = np.array([[[8, 8, 3],[4, 4, 4]]])
###d = np.array([[6, 6, 6]])
##c = np.append(a, b, axis=0)
###c = np.append(c, d, axis=0)
#
#
#import tensorflow as tf
#
#
#rconv1_w = tf.Variable(tf.random_normal([11,11,3,96]), name="w1")
#rconv1_b = tf.Variable(tf.random_normal([96]), name="b1")
#
#abs2 = tf.abs(rconv1_w)
#
#with tf.Session() as sess:
#    saver = tf.train.import_meta_graph('../model/fcann_v1.ckpt-2.meta')
#    init = tf.global_variables_initializer()
#    sess.run(init)
#    
#    
#    bares = sess.run(abs2)
#    
#    r = saver.restore(sess, '../model/fcann_v1.ckpt-2')   
##    all_vars = tf.get_collection('w1')
##    
##    for v in all_vars:
##        v_ = sess.run(v)
##        print(v_)
#    res = sess.run(rconv1_w.assign(tf.get_collection('w1')[0]))
#    ares = sess.run(abs2)
#   
#
#
##v1 = tf.Variable(tf.zeros([2, 2], dtype=tf.float32, name='v1'))
##saver = tf.train.Saver()
##
##with tf.Session() as sess:
##    sess.run(tf.global_variables_initializer())
##    print(sess.run(v1))
##    save_path = saver.save(sess, './model.ckpt')
##    print("model saved in file:", save_path)
##
##    # Create an op to increment v1, run it, and print the result.   
##    increment_op = v1.assign_add(tf.ones([2, 2]))
##    sess.run(increment_op)
##    print(sess.run(v1))
##
##    # Restore from the checkpoint saved above.
##    saver.restore(sess, './model.ckpt')
##    print(sess.run(v1))