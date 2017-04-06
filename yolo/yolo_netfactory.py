import tensorflow as tf
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from scipy import misc
#import time


alpha = 0.1

def conv(input_src , weight, bias, step, padding='SAME'):
    
     conv = tf.nn.conv2d(input_src, weight, strides=[1, step, step, 1], padding=padding)
     conv_biased = tf.add(conv ,bias)	
     return tf.maximum(alpha*conv_biased,conv_biased)
 
def fc_layer(input_src, weight, bias, flat = False,linear = False):
    
    input_shape = input_src.get_shape().as_list()
    if flat:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs_transposed = tf.transpose(input_src,(0,3,1,2))
        inputs_processed = tf.reshape(inputs_transposed, [-1,dim])
    else:
        dim = input_shape[1]
        inputs_processed = input_src
        
    if linear : return tf.add(tf.matmul(inputs_processed,weight),bias)
    
    ip = tf.add(tf.matmul(inputs_processed,weight),bias)
    return tf.maximum(alpha*ip,ip)

def pooling_layer(inputs,size,stride):
		return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME')

def yolo_vanilla_train(data, ds_yolo, keep_prob):
    
     with tf.name_scope("yolo_vanilla_train"):
         
        with tf.name_scope("conv1"):
            conv1 = conv(data, ds_yolo['conv1w'], ds_yolo['conv1b'],1)
            
        with tf.name_scope("pool1"):
            pool1 = pooling_layer(conv1,2,2)
            
        with tf.name_scope("conv2"):
            conv2 = conv(pool1, ds_yolo['conv2w'], ds_yolo['conv2b'],1)
            
        with tf.name_scope("pool2"):
            pool2 = pooling_layer(conv2,2,2)
            
        with tf.name_scope("conv3"):
            conv3 = conv(pool2, ds_yolo['conv3w'], ds_yolo['conv3b'],1)
            
        with tf.name_scope("pool3"):
            pool3 = pooling_layer(conv3,2,2)
            
        with tf.name_scope("conv4"):
            conv4 = conv(pool3, ds_yolo['conv4w'], ds_yolo['conv4b'],1)
            
        with tf.name_scope("pool4"):
            pool4 = pooling_layer(conv4,2,2)
            
        with tf.name_scope("conv5"):
            conv5 = conv(pool4, ds_yolo['conv5w'], ds_yolo['conv5b'],1)
        
        with tf.name_scope("pool5"):
            pool5 = pooling_layer(conv5,2,2)
            
        with tf.name_scope("conv6"):
            conv6 = conv(pool5, ds_yolo['conv6w'], ds_yolo['conv6b'],1)
            
        with tf.name_scope("pool6"):
            pool6 = pooling_layer(conv6,2,2)
            
        with tf.name_scope("conv7"):
            conv7 = conv(pool6, ds_yolo['conv7w'], ds_yolo['conv7b'],1)
        
        with tf.name_scope("conv8"):
            conv8 = conv(conv7, ds_yolo['conv8w'], ds_yolo['conv8b'],1)
        
        with tf.name_scope("conv9"):
            conv9 = conv(conv8, ds_yolo['conv9w'], ds_yolo['conv9b'],1)
            
        with tf.name_scope("fc10"):
            fc10 = fc_layer(conv9, ds_yolo['fc10w'], ds_yolo['fc10b'],flat=True,linear=False)
            
        with tf.name_scope("dropout1"):
            dropout1 = tf.nn.dropout(fc10, keep_prob)
        
        with tf.name_scope("fc11"):
            fc11 = fc_layer(dropout1, ds_yolo['fc11w'], ds_yolo['fc11b'],flat=False,linear=False)
        
        with tf.name_scope("fc12"):
            fc12 = fc_layer(fc11, ds_yolo['fc12w'], ds_yolo['fc12b'],flat=False,linear=True)
     return fc12


def yolo_vanilla(data, ds_yolo, keep_prob):
    
     with tf.name_scope("yolo_vanilla_test"):
         
        with tf.name_scope("conv1"):
            conv1 = conv(data, ds_yolo['conv1w'], ds_yolo['conv1b'],1)
            
        with tf.name_scope("pool1"):
            pool1 = pooling_layer(conv1,2,2)
            
        with tf.name_scope("conv2"):
            conv2 = conv(pool1, ds_yolo['conv2w'], ds_yolo['conv2b'],1)
            
        with tf.name_scope("pool2"):
            pool2 = pooling_layer(conv2,2,2)
            
        with tf.name_scope("conv3"):
            conv3 = conv(pool2, ds_yolo['conv3w'], ds_yolo['conv3b'],1)
            
        with tf.name_scope("pool3"):
            pool3 = pooling_layer(conv3,2,2)
            
        with tf.name_scope("conv4"):
            conv4 = conv(pool3, ds_yolo['conv4w'], ds_yolo['conv4b'],1)
            
        with tf.name_scope("pool4"):
            pool4 = pooling_layer(conv4,2,2)
            
        with tf.name_scope("conv5"):
            conv5 = conv(pool4, ds_yolo['conv5w'], ds_yolo['conv5b'],1)
        
        with tf.name_scope("pool5"):
            pool5 = pooling_layer(conv5,2,2)
            
        with tf.name_scope("conv6"):
            conv6 = conv(pool5, ds_yolo['conv6w'], ds_yolo['conv6b'],1)
            
        with tf.name_scope("pool6"):
            pool6 = pooling_layer(conv6,2,2)
            
        with tf.name_scope("conv7"):
            conv7 = conv(pool6, ds_yolo['conv7w'], ds_yolo['conv7b'],1)
        
        with tf.name_scope("conv8"):
            conv8 = conv(conv7, ds_yolo['conv8w'], ds_yolo['conv8b'],1)
        
        with tf.name_scope("conv9"):
            conv9 = conv(conv8, ds_yolo['conv9w'], ds_yolo['conv9b'],1)
            
        with tf.name_scope("fc10"):
            fc10 = fc_layer(conv9, ds_yolo['fc10w'], ds_yolo['fc10b'],flat=True,linear=False)
            
        with tf.name_scope("dropout1"):
            dropout1 = tf.nn.dropout(fc10, keep_prob)
        
        with tf.name_scope("fc11"):
            fc11 = fc_layer(dropout1, ds_yolo['fc11w'], ds_yolo['fc11b'],flat=False,linear=False)
        
        with tf.name_scope("fc12"):
            fc12 = fc_layer(fc11, ds_yolo['fc12w'], ds_yolo['fc12b'],flat=False,linear=True)

     return fc12

def yolo_ds_train(data, ds_yolo, keep_prob):
    
    with tf.name_scope("ds_yolo_train"):
        with tf.name_scope("conv1"):
            conv1 = conv(data, ds_yolo['conv1w'], ds_yolo['conv1b'],2)
            
        with tf.name_scope("conv2"):
            conv2 = conv(conv1, ds_yolo['conv2w'], ds_yolo['conv2b'],2)
            
        with tf.name_scope("conv3"):
            conv3 = conv(conv2, ds_yolo['conv3w'], ds_yolo['conv3b'],2)
            
        with tf.name_scope("conv4"):
            conv4 = conv(conv3, ds_yolo['conv4w'], ds_yolo['conv4b'],2)
            
        with tf.name_scope("conv5"):
            conv5 = conv(conv4, ds_yolo['conv5w'], ds_yolo['conv5b'],2)
            
        with tf.name_scope("conv6"):
            conv6 = conv(conv5, ds_yolo['conv6w'], ds_yolo['conv6b'],2)
            
        with tf.name_scope("conv7"):
            conv7 = conv(conv6, ds_yolo['conv7w'], ds_yolo['conv7b'],1)
        
        with tf.name_scope("conv8"):
            conv8 = conv(conv7, ds_yolo['conv8w'], ds_yolo['conv8b'],1)
        
        with tf.name_scope("conv9"):
            conv9 = conv(conv8, ds_yolo['conv9w'], ds_yolo['conv9b'],1)
            
        with tf.name_scope("fc10"):
            fc10 = fc_layer(conv9, ds_yolo['fc10w'], ds_yolo['fc10b'],flat=True,linear=False)
            
        with tf.name_scope("dropout1"):
            dropout1 = tf.nn.dropout(fc10, keep_prob)
        
        with tf.name_scope("fc11"):
            fc11 = fc_layer(dropout1, ds_yolo['fc11w'], ds_yolo['fc11b'],flat=False,linear=False)
    
        with tf.name_scope("dropout2"):
            dropout2 = tf.nn.dropout(fc11, keep_prob)
        
        with tf.name_scope("fc12"):
            fc12 = fc_layer(dropout2, ds_yolo['fc12w'], ds_yolo['fc12b'],flat=False,linear=True)


    return fc12

def yolo_ds(data, ds_yolo, keep_prob):
    
    with tf.name_scope("ds_yolo"):
        with tf.name_scope("conv1"):
            conv1 = conv(data, ds_yolo['conv1w'], ds_yolo['conv1b'],2)
            
        with tf.name_scope("conv2"):
            conv2 = conv(conv1, ds_yolo['conv2w'], ds_yolo['conv2b'],2)
            
        with tf.name_scope("conv3"):
            conv3 = conv(conv2, ds_yolo['conv3w'], ds_yolo['conv3b'],2)
            
        with tf.name_scope("conv4"):
            conv4 = conv(conv3, ds_yolo['conv4w'], ds_yolo['conv4b'],2)
            
        with tf.name_scope("conv5"):
            conv5 = conv(conv4, ds_yolo['conv5w'], ds_yolo['conv5b'],2)
            
        with tf.name_scope("conv6"):
            conv6 = conv(conv5, ds_yolo['conv6w'], ds_yolo['conv6b'],2)
            
        with tf.name_scope("conv7"):
            conv7 = conv(conv6, ds_yolo['conv7w'], ds_yolo['conv7b'],1)
        
        with tf.name_scope("conv8"):
            conv8 = conv(conv7, ds_yolo['conv8w'], ds_yolo['conv8b'],1)
        
        with tf.name_scope("conv9"):
            conv9 = conv(conv8, ds_yolo['conv9w'], ds_yolo['conv9b'],1)
            
        with tf.name_scope("fc10"):
            fc10 = fc_layer(conv9, ds_yolo['fc10w'], ds_yolo['fc10b'],flat=True,linear=False)
        
        with tf.name_scope("fc11"):
            fc11 = fc_layer(fc10, ds_yolo['fc11w'], ds_yolo['fc11b'],flat=False,linear=False)

        with tf.name_scope("fc12"):
            fc12 = fc_layer(fc11, ds_yolo['fc12w'], ds_yolo['fc12b'],flat=False,linear=True)


    return fc12