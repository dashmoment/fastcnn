import numpy as np
import tensorflow as tf
import random

import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from nets import custom_layers

img_shape=(300, 300)
num_classes=21
no_annotation_label=21
feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
anchor_size_bounds=[0.15, 0.90]
anchor_sizes=[(21., 45.),
              (45., 99.),
              (99., 153.),
              (153., 207.),
              (207., 261.),
              (261., 315.)]

anchor_ratios=[[2, .5],
               [2, .5, 3, 1./3],
               [2, .5, 3, 1./3],
               [2, .5, 3, 1./3],
               [2, .5],
               [2, .5]]
anchor_steps=[8, 16, 32, 64, 100, 300]
anchor_offset=0.5
normalizations=[20, -1, -1, -1, -1, -1]
prior_scaling=[0.1, 0.1, 0.2, 0.2]
prediction_fn=slim.softmax

variable_names = [
            'conv1/conv1_1/weights',
            'conv1/conv1_1/biases',
            'conv1/conv1_2/weights',
            'conv1/conv1_2/biases',
            'conv2/conv2_1/weights',
            'conv2/conv2_1/biases',
            'conv2/conv2_2/weights',
            'conv2/conv2_2/biases',
            'conv3/conv3_1/weights',
            'conv3/conv3_1/biases',
            'conv3/conv3_2/weights',
            'conv3/conv3_2/biases',
            'conv3/conv3_3/weights',
            'conv3/conv3_3/biases',
            'conv4/conv4_1/weights',
            'conv4/conv4_1/biases',
            'conv4/conv4_2/weights',
            'conv4/conv4_2/biases',
            'conv4/conv4_3/weights',
            'conv4/conv4_3/biases',
            'conv5/conv5_1/weights',
            'conv5/conv5_1/biases',
            'conv5/conv5_2/weights',
            'conv5/conv5_2/biases',
            'conv5/conv5_3/weights',
            'conv5/conv5_3/biases',
            'conv6/weights',
            'conv6/biases',
            'conv7/weights',
            'conv7/biases',
            'block8/conv1x1/weights',
            'block8/conv1x1/biases',
            'block8/conv3x3/weights',
            'block8/conv3x3/biases',
            'block9/conv1x1/weights',
            'block9/conv1x1/biases',
            'block9/conv3x3/weights',
            'block9/conv3x3/biases',
            'block10/conv1x1/weights',
            'block10/conv1x1/biases',
            'block10/conv3x3/weights',
            'block10/conv3x3/biases',
            'block11/conv1x1/weights',
            'block11/conv1x1/biases',
            'block11/conv3x3/weights',
            'block11/conv3x3/biases'
        ]

class ssd_shrink_network:
    
    
    def __init__(self, scope,  ratio, ckpt_filename = '',gpu = '/gpu:1'):
        
        self.net_shape = (300, 300)
        self.data_format = 'NHWC'
        self.dropout_keep_prob = 0.5
        self.reuse = None
        self.is_training = True
        
        self.gpu = gpu
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
        self.ckpt_filename = ckpt_filename
        
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        self.glabel = tf.placeholder(tf.int64)
        self.glocation = tf.placeholder(tf.float32)
        self.gscore = tf.placeholder(tf.float32)
        
        self.creat_network(scope, ratio)
        
#        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=''):
#            print (i.name)
    
  
        
    def creat_network(self, scope, ratio):
        
        with tf.device(self.gpu):
        
            self.ssd_net = ssd_vgg_300.SSDNet()
            
            image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            self.img_input, None, None, self.net_shape, self.data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
            self.image_4d = tf.expand_dims(image_pre, 0)
        
            with slim.arg_scope(self.ssd_net.arg_scope(data_format=self.data_format)):
                end_points = {}
                with tf.variable_scope(scope,scope,[self.image_4d], reuse=self.reuse):
                    # Original VGG-16 blocks.
                    net = slim.repeat(self.image_4d, 2, slim.conv2d, int(64*ratio), [3, 3], scope='conv1')
                    end_points['block1'] = net
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    # Block 2.
                    net = slim.repeat(net, 2, slim.conv2d, int(128*ratio), [3, 3], scope='conv2')
                    end_points['block2'] = net
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    # Block 3.
                    net = slim.repeat(net, 3, slim.conv2d, int(256*ratio), [3, 3], scope='conv3')
                    end_points['block3'] = net
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    # Block 4.
                    net = slim.repeat(net, 3, slim.conv2d, int(512*ratio), [3, 3], scope='conv4')
                    end_points['block4'] = net
                    net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    # Block 5.
                    net = slim.repeat(net, 3, slim.conv2d, int(512*ratio), [3, 3], scope='conv5')
                    end_points['block5'] = net
                    net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')
                
                    # Additional SSD blocks.
                    # Block 6: let's dilate the hell out of it!
                    net = slim.conv2d(net, int(1024*ratio), [3, 3], rate=6, scope='conv6')
                    end_points['block6'] = net
                    net = tf.layers.dropout(net, rate=self.dropout_keep_prob, training=self.is_training)
                    # Block 7: 1x1 conv. Because the fuck.
                    net = slim.conv2d(net, int(1024*ratio), [1, 1], scope='conv7')
                    end_points['block7'] = net
                    net = tf.layers.dropout(net, rate=self.dropout_keep_prob, training=self.is_training)
                
                    # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
                    end_point = 'block8'
                    with tf.variable_scope(end_point):
                        net = slim.conv2d(net, int(256*ratio), [1, 1], scope='conv1x1')
                        net = custom_layers.pad2d(net, pad=(1, 1))
                        net = slim.conv2d(net, int(512*ratio), [3, 3], stride=2, scope='conv3x3', padding='VALID')
                    end_points[end_point] = net
                    end_point = 'block9'
                    with tf.variable_scope(end_point):
                        net = slim.conv2d(net, int(128*ratio), [1, 1], scope='conv1x1')
                        net = custom_layers.pad2d(net, pad=(1, 1))
                        net = slim.conv2d(net, int(256*ratio), [3, 3], stride=2, scope='conv3x3', padding='VALID')
                    end_points[end_point] = net
                    end_point = 'block10'
                    with tf.variable_scope(end_point):
                        net = slim.conv2d(net, int(128*ratio), [1, 1], scope='conv1x1')
                        net = slim.conv2d(net, int(256*ratio), [3, 3], scope='conv3x3', padding='VALID')
                    end_points[end_point] = net
                    end_point = 'block11'
                    with tf.variable_scope(end_point):
                        net = slim.conv2d(net, int(128*ratio), [1, 1], scope='conv1x1')
                        net = slim.conv2d(net, int(256*ratio), [3, 3], scope='conv3x3', padding='VALID')
                    end_points[end_point] = net
                    
                    # Prediction and localisations layers.
                    self.predictions = []
                    self.logits = []
                    self.localisations = []
                    for i, layer in enumerate(self.ssd_net.params.feat_layers):
                        with tf.variable_scope(layer + '_box'):
                            p, l = ssd_vgg_300.ssd_multibox_layer(end_points[layer],
                                                      num_classes,
                                                      anchor_sizes[i],
                                                      anchor_ratios[i],
                                                      normalizations[i])
                        self.predictions.append(prediction_fn(p))
                        self.logits.append(p)
                        self.localisations.append(l)
                        
        
        self.loss = ssd_losses(self.logits, self.localisations, self.glabel, self.glocation, self.gscore)
        self.solver = tf.train.MomentumOptimizer(learning_rate = 0.8, momentum=0.9).minimize(self.loss)
        
            
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())
                    
        if self.ckpt_filename != '':
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.ckpt_filename)
            
    def train_op(self, img, glabel, glocation, gscore):
        
        fglabel, fglocation, fgscore = self.flatten_output(glabel, glocation, gscore)
        
        self.sess.run(self.loss,  feed_dict={self.img_input: img,self.glabel:fglabel, self.glocation:fglocation, self.gscore:fgscore})

    def flatten_output(self, glabel, glocation, gscore):

        # Flatten out all vectors!
        
        fgclasses = []
        fgscores = []
        fglocalisations = []
        for i in range(len(self.logits)):
            
            fgclasses.append(tf.reshape(glabel[i], [-1]))
            fgscores.append(tf.reshape(gscore[i], [-1]))
            
            fglocalisations.append(tf.reshape(glocation[i], [-1, 4]))
            
        # And concat the crap!
        
        self.gclasses = self.sess.run(tf.concat(fgclasses, axis=0))
        self.gscores = self.sess.run(tf.concat(fgscores, axis=0))
        self.glocalisations = self.sess.run(tf.concat(fglocalisations, axis=0))
        
        return self.gclasses, self.gscores, self.glocalisations
        
                
        
        
        
def model_prunning(var_scop, s_varscop, v_obj, s_obj):
    
    for var in variable_names:
        
        with tf.variable_scope(var_scop) as scope:
            scope.reuse_variables()
            var_ori = tf.get_variable(var)
        
        
        with tf.variable_scope(s_varscop) as scope:
            scope.reuse_variables()
            var_s = tf.get_variable(var)
            
        tensors = v_obj.sess.run(var_ori)
        tensors_s = s_obj.sess.run(var_s) 
        
        if np.shape(tensors) != np.shape(tensors_s):
             
             if (len(np.shape(tensors))) > 1:
            
                if (len(np.shape(tensors))) > 2:     
        
                    dim_array = np.zeros((np.shape(tensors_s)[0],np.shape(tensors_s)[1],np.shape(tensors_s)[2],np.shape(tensors)[3]), dtype=np.float32)                    
                    dim_sum = np.sum(np.sum(tensors, axis=0),axis=0)
                    dim_sum = np.abs(dim_sum)
                    
                    del_axis = []
                    
                    for i in range(dim_sum.shape[1]):
                        axis_sum = dim_sum[:,i]
                        axis_sort = np.unravel_index(axis_sum.argsort(axis=None), dims=int(len(axis_sum)))
                        del_axis = [axis_sort[0][x] for x in range(np.shape(tensors)[-2] -  np.shape(tensors_s)[-2])]
                        tmp = tensors[:,:,:,i]
                        stmp = np.delete(tmp,del_axis, -1)
                        dim_array[:,:,:,i] = stmp
                                           
                else:  
                     
                     dim_sum = np.sum(tensors, axis=1)
                     dim_sum = np.abs(dim_sum)
                     dim_sort = np.unravel_index(dim_sum.argsort(axis=None), dims=int(len(dim_sum)))
                     del_list = [dim_sort[0][x] for x in range(np.shape(tensors)[-2] -  np.shape(tensors_s)[-2])]
                     dim_array = np.delete(tensors,del_list, -2)
                     
                     
                if (len(np.shape(tensors))) > 2:        
                        kernel_sum = np.sum(np.sum(np.sum(dim_array,axis=0), axis=0),axis=0)
                else:
                        kernel_sum = np.sum(dim_array, axis=0)
                    
                kernel_sum = np.abs(kernel_sum)
                kernel_sort = np.unravel_index(kernel_sum.argsort(axis=None), dims=int(len(kernel_sum)))
                kdel_list = [kernel_sort[0][x] for x in range(np.shape(tensors)[-1] -  np.shape(tensors_s)[-1])]
                kernel_array = np.delete(dim_array, kdel_list, -1)
                
             else:
                kernel_array = np.delete(tensors, kdel_list,0)
                
           
        else:   
             kernel_array = tensors
    
    
        with tf.variable_scope(s_varscop) as scope:
            scope.reuse_variables()
            op = tf.assign(tf.get_variable(var)  , kernel_array)  
            s_obj.sess.run(op)
            

       
        
def ssd_losses(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               device='/gpu:1',
               scope=None):
    with tf.name_scope(scope, 'ssd_losses'):
        lshape = tf.get_shape(logits[0], 5)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []     
        flocalisations = []
       
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))            
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)   
        localisations = tf.concat(flocalisations, axis=0)     
        dtype = logits.dtype

        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        n_positives = tf.reduce_sum(fpmask)

        # Hard negative mining...
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)
        nmask = tf.logical_and(tf.logical_not(pmask),
                               gscores > -0.5)
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=gclasses)
            loss1 = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)
            loss2 = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = custom_layers.abs_smooth(localisations - glocalisations)
            loss3 = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
            tf.losses.add_loss(loss)
        loss = loss1 + loss2 + loss3 
        return loss        
        
        
        
        
        
        
        
        
        
        