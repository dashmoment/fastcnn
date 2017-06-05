#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:43:39 2017

@author: ubuntu
"""

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
from scipy.misc import imread
from scipy.misc import imresize
from caffe_classes import class_names
from PIL import Image
import matplotlib.pyplot as plt
import time



def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


def vanilla_alex_full(data):

    
    net_data = np.load("../model/bvlc_alexnet.npy", encoding='latin1').item()
    
    with tf.name_scope("Vanilla_Alex"): 
        s_h = 4; s_w = 4
        conv1W = tf.Variable(net_data["conv1"][0], trainable = False)
        conv1b = tf.Variable(net_data["conv1"][1], trainable = False)
    
        conv1_in = tf.nn.conv2d(data, conv1W, strides=[1,s_h,s_w,1], padding='SAME')
        conv1_add = tf.nn.bias_add(conv1_in, conv1b)
        conv1 = tf.nn.relu(conv1_add)
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
        k_h = 3; k_w = 3; s_h = 2; s_w = 2
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID')
            
        #conv2
        #conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv2W = tf.Variable(net_data["conv2"][0], trainable = False)
        conv2b = tf.Variable(net_data["conv2"][1], trainable = False)       
        conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
        k_h = 3; k_w = 3; s_h = 2; s_w = 2
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID')
        
       
        #conv3
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        conv3W = tf.Variable(net_data["conv3"][0], trainable = False)
        conv3b = tf.Variable(net_data["conv3"][1], trainable = False)
        conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)
        
        #conv4
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        conv4W = tf.Variable(net_data["conv4"][0], trainable = False)
        conv4b = tf.Variable(net_data["conv4"][1], trainable = False)
        conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)
        
        
        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5W = tf.Variable(net_data["conv5"][0], trainable = False)
        conv5b = tf.Variable(net_data["conv5"][1], trainable = False)
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)
        
        #maxpool5
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        
        #fc6
        #fc(4096, name='fc6')
        fc6W = tf.Variable(net_data["fc6"][0], trainable = False)
        fc6b = tf.Variable(net_data["fc6"][1], trainable = False)
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
        
        #fc7
        #fc(4096, name='fc7')
        fc7W = tf.Variable(net_data["fc7"][0], trainable = False)
        fc7b = tf.Variable(net_data["fc7"][1], trainable = False)
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
        
        #fc8
        #fc(1000, relu=False, name='fc8')
        fc8W = tf.Variable(net_data["fc8"][0], trainable = False)
        fc8b = tf.Variable(net_data["fc8"][1], trainable = False)
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)     
        prob = tf.nn.softmax(fc8)
        
        
        
    
        return fc8W,fc8b, prob

def mini_alex_full(data,conv1W,conv1b):
    
     with tf.name_scope("Common"): 
         
        net_data = np.load("../model/bvlc_alexnet.npy", encoding='latin1').item()
        s_h = 4; s_w = 4
    
        conv1_in = tf.nn.conv2d(data, conv1W, strides=[1,s_h,s_w,1], padding='SAME')
        conv1_add = tf.nn.bias_add(conv1_in, conv1b)
        conv1 = tf.nn.relu(conv1_add)
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
        k_h = 3; k_w = 3; s_h = 2; s_w = 2
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID')
            
        #conv2
        #conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv2W = tf.Variable(net_data["conv2"][0])
        conv2b = tf.Variable(net_data["conv2"][1])
        
        conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
       
        conv2_add = tf.nn.bias_add(conv2_in, conv2b)
        conv2 = tf.nn.relu(conv2_add)
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
        k_h = 3; k_w = 3; s_h = 2; s_w = 2
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID')
        
        
        #conv3
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        conv3W = tf.Variable(net_data["conv3"][0])
        conv3b = tf.Variable(net_data["conv3"][1])
        conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)
        
        #conv4
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        conv4W = tf.Variable(net_data["conv4"][0])
        conv4b = tf.Variable(net_data["conv4"][1])
        conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)
        
        
        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5W = tf.Variable(net_data["conv5"][0])
        conv5b = tf.Variable(net_data["conv5"][1])
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)
        
        #maxpool5
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        
        #fc6
        #fc(4096, name='fc6')
        fc6W = tf.Variable(net_data["fc6"][0])
        fc6b = tf.Variable(net_data["fc6"][1])
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
        
        #fc7
        #fc(4096, name='fc7')
        fc7W = tf.Variable(net_data["fc7"][0])
        fc7b = tf.Variable(net_data["fc7"][1])
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
        
        #fc8
        #fc(1000, relu=False, name='fc8')
        fc8W = tf.Variable(net_data["fc8"][0])
        fc8b = tf.Variable(net_data["fc8"][1])
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
        
        
        #prob
        #softmax(name='prob'))
        prob = tf.nn.softmax(fc8)
        return prob

def full_alex_test(data):

    with tf.name_scope("conv1"):
        s_h = 8; s_w = 8
        rconv1W = tf.Variable(tf.random_normal([11,11,3,96],stddev=0.01))
        rconv1b = tf.Variable(tf.random_normal([96],mean= 0,stddev= 0.01)) 
        conv1_in = tf.nn.conv2d(data, rconv1W, strides=[1,s_h,s_w,1], padding='VALID')
        conv1_add = tf.nn.bias_add(conv1_in, rconv1b)
        conv1 = tf.nn.relu(conv1_add)
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)
        
    #conv2
    with tf.name_scope("conv2"):
        k_h = 5; k_w = 5; c_o = 256; s_h = 2; s_w = 2
        rconv2W = tf.Variable(tf.random_normal([k_h,k_w,96,256],stddev=0.01))
        rconv2b = tf.Variable(tf.random_normal([256],mean= 0,stddev= 0.01))        
        conv2_in = tf.nn.conv2d(lrn1, rconv2W, strides=[1,s_h,s_w,1], padding='VALID')
        conv2_add = tf.nn.bias_add(conv2_in, rconv2b)
        conv2 = tf.nn.relu(conv2_add)
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #conv3
    with tf.name_scope("conv3"):
    #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1
        rconv3W = tf.Variable(tf.random_normal([k_h,k_w,256,c_o],stddev=0.01))
        rconv3b = tf.Variable(tf.random_normal([384],mean= 0,stddev= 0.01))
        conv3_in = tf.nn.conv2d(lrn2, rconv3W, strides=[1,s_h,s_w,1], padding='SAME')
        conv3_add = tf.nn.bias_add(conv3_in, rconv3b)
        conv3 = tf.nn.relu(conv3_add)
    
    #conv4
    with tf.name_scope("conv4"):
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1
        rconv4W = tf.Variable(tf.random_normal([k_h,k_w,384,c_o],stddev=0.01))
        rconv4b = tf.Variable(tf.random_normal([384],mean= 0,stddev= 0.01))
        conv4_in = tf.nn.conv2d(conv3, rconv4W, strides=[1,s_h,s_w,1], padding='SAME')
        conv4_add = tf.nn.bias_add(conv4_in, rconv4b)
        conv4 = tf.nn.relu(conv4_add)
    
    
    #conv5
    with tf.name_scope("conv5"):
        k_h = 3; k_w = 3; c_o = 256; s_h = 2; s_w = 2
        rconv5W = tf.Variable(tf.random_normal([k_h,k_w,384,c_o],stddev=0.01))
        rconv5b = tf.Variable(tf.random_normal([c_o],mean= 0,stddev= 0.01))
        conv5_in = tf.nn.conv2d(conv4, rconv5W, strides=[1,s_h,s_w,1], padding='VALID')
        conv5_add = tf.nn.bias_add(conv5_in, rconv5b)
        conv5 = tf.nn.relu(conv5_add)


    #fc6
    #fc(4096, name='fc6')
    with tf.name_scope("fc6"):
        len_input = int(np.prod(conv5.get_shape()[1:]))
        
        rfc6W = tf.Variable(tf.random_normal([len_input, 4096], stddev=0.01))
        rfc6b = tf.Variable(tf.random_normal([4096], mean= 0,stddev= 0.01))
        fc6 = tf.nn.relu_layer(tf.reshape(conv5, [-1, len_input]), rfc6W, rfc6b)
    
    #fc7
    #fc(4096, name='fc7')
    with tf.name_scope("fc7"):
        rfc7W = tf.Variable(tf.random_normal([4096, 4096], stddev=0.01))
        rfc7b = tf.Variable(tf.random_normal([4096], mean= 0,stddev= 0.01))
        fc7 = tf.nn.relu_layer(fc6, rfc7W, rfc7b)
    
    #fc8
    #fc(1000, relu=False, name='fc8')
    with tf.name_scope("fc8"):
        rfc8W = tf.Variable(tf.random_normal([4096, 1000], stddev=0.01))
        rfc8b = tf.Variable(tf.random_normal([1000], mean= 0,stddev= 0.01))
        fc8 = tf.nn.xw_plus_b(fc7, rfc8W, rfc8b)     
        prob = tf.nn.softmax(fc8)

    return prob

def full_alex_ds(data, model):

    with tf.name_scope("Mini"):

        with tf.name_scope("conv1"):
            s_h = 8; s_w = 8
            rconv1W = model['conv1w']
            rconv1b = model['conv1b']
            conv1_in = tf.nn.conv2d(data,rconv1W, strides=[1,s_h,s_w,1], padding='VALID')
            conv1_add = tf.nn.bias_add(conv1_in, rconv1b)
            conv1 = tf.nn.relu(conv1_add)
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
            lrn1 = tf.nn.local_response_normalization(conv1,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
        #conv2
        with tf.name_scope("conv2"):
            k_h = 5; k_w = 5; c_o = 256; s_h = 2; s_w = 2
            rconv2W = model['conv2w']
            rconv2b = model['conv2b']        
            conv2_in = tf.nn.conv2d(lrn1, rconv2W, strides=[1,s_h,s_w,1], padding='VALID')
            conv2_add = tf.nn.bias_add(conv2_in, rconv2b)
            conv2 = tf.nn.relu(conv2_add)
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
            lrn2 = tf.nn.local_response_normalization(conv2,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)

        #conv3
        with tf.name_scope("conv3"):
        #conv(3, 3, 384, 1, 1, name='conv3')
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1
            rconv3W = model['conv3w']
            rconv3b = model['conv3b']
            conv3_in = tf.nn.conv2d(lrn2, rconv3W, strides=[1,s_h,s_w,1], padding='SAME')
            conv3_add = tf.nn.bias_add(conv3_in, rconv3b)
            conv3 = tf.nn.relu(conv3_add)
        
        #conv4
        with tf.name_scope("conv4"):
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1
            rconv4W = model['conv4w']
            rconv4b = model['conv4b']
            conv4_in = tf.nn.conv2d(conv3, rconv4W, strides=[1,s_h,s_w,1], padding='SAME')
            conv4_add = tf.nn.bias_add(conv4_in, rconv4b)
            conv4 = tf.nn.relu(conv4_add)
        
        
        #conv5
        with tf.name_scope("conv5"):
            k_h = 3; k_w = 3; c_o = 256; s_h = 2; s_w = 2
            rconv5W = model['conv5w']
            rconv5b = model['conv5b']
            conv5_in = tf.nn.conv2d(conv4, rconv5W, strides=[1,s_h,s_w,1], padding='VALID')
            conv5_add = tf.nn.bias_add(conv5_in, rconv5b)
            conv5 = tf.nn.relu(conv5_add)


        #fc6
        #fc(4096, name='fc6')
        with tf.name_scope("fc6"):
            len_input = int(np.prod(conv5.get_shape()[1:]))
            
            rfc6W = model['fc6w']
            rfc6b = model['fc6b']
            fc6 = tf.nn.relu_layer(tf.reshape(conv5, [-1, len_input]), rfc6W, rfc6b)
        
        #fc7
        #fc(4096, name='fc7')
        with tf.name_scope("fc7"):
            rfc7W = model['fc7w']
            rfc7b = model['fc7b']
            fc7 = tf.nn.relu_layer(fc6, rfc7W, rfc7b)
        
        #fc8
        #fc(1000, relu=False, name='fc8')
        with tf.name_scope("fc8"):
            rfc8W = model['fc8w']
            rfc8b = model['fc8b']
            fc8 = tf.nn.xw_plus_b(fc7, rfc8W, rfc8b)     
            prob = tf.nn.softmax(fc8)
            

    return prob


def mini_alex_ds(data,conv1W,conv1b):
    net_data = np.load("../model/bvlc_alexnet.npy", encoding='latin1').item()
    s_h = 8; s_w = 8

    conv1_in = tf.nn.conv2d(data, conv1W, strides=[1,s_h,s_w,1], padding='VALID')
    conv1_add = tf.nn.bias_add(conv1_in, conv1b)
    conv1 = tf.nn.relu(conv1_add)
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
#    k_h = 3; k_w = 3; s_h = 2; s_w = 2
#    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID')
        
    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    
    conv2_in = conv(lrn1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
   
    conv2_add = tf.nn.bias_add(conv2_in, conv2b)
    conv2 = tf.nn.relu(conv2_add)
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
    k_h = 3; k_w = 3; s_h = 2; s_w = 2
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID')
    
    
    return maxpool2




    
    
    

    
        
    
    
    
    
    
    
    
    
    
    
    