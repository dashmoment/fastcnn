import numpy as np
import tensorflow as tf
import random

import tf_extended as tfe
import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from nets import custom_layers

import vanilla_ssd as van

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

            'block4_box/conv_loc/weights',
            'block4_box/conv_loc/biases',
            'block4_box/conv_cls/weights',
            'block4_box/conv_cls/biases',
            'block7_box/conv_loc/weights',
            'block7_box/conv_loc/biases',
            'block7_box/conv_cls/weights',
            'block7_box/conv_cls/biases',
            'block8_box/conv_loc/weights',
            'block8_box/conv_loc/biases',
            'block8_box/conv_cls/weights',
            'block8_box/conv_cls/biases',
            'block9_box/conv_loc/weights',
            'block9_box/conv_loc/biases',
            'block9_box/conv_cls/weights',
            'block9_box/conv_cls/biases',
            'block10_box/conv_loc/weights',
            'block10_box/conv_loc/biases',
            'block10_box/conv_cls/weights',
            'block10_box/conv_cls/biases',
            'block11_box/conv_loc/weights',
            'block11_box/conv_loc/biases',
            'block11_box/conv_cls/weights',
            'block11_box/conv_cls/biases',
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
            'block4_box/L2Normalization/gamma',
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
    
    
    def __init__(self, scope,  ratio, batch_size = 64 , ckpt_filename = '',gpu = '/gpu:0',reuse=None):
        
        self.van = van.vanilla_ssd_net('/gpu:0', reuse=reuse)
        
        self.net_shape = (300, 300)
        self.data_format = 'NHWC'
        self.dropout_keep_prob = 0.5
        self.reuse = reuse
        self.is_training = True
        self.scope = scope
        
        self.batch_size = batch_size
        
        self.gpu = gpu
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
        self.ckpt_filename = ckpt_filename
        
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.net_shape[0], self.net_shape[1], 3))
        self.glabel = tf.placeholder(tf.int64)
        self.glocation = tf.placeholder(tf.float32)
        self.gscore = tf.placeholder(tf.float32)
        
        self.creat_network(scope, ratio)
        
        
#        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=''):
#            print (i.name)
    
    def model_pruning(self):
                    
        model_prunning('ssd_300_vgg', self.scope, self.van, self)
        
    def creat_network(self, scope, ratio):
         
        with tf.device(self.gpu):
        
            self.ssd_net = ssd_vgg_300.SSDNet()

            self.ssd_anchors = self.ssd_net.anchors(self.net_shape)
            
            image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            self.img_input, None, None, self.net_shape, self.data_format)
            self.image_4d = tf.expand_dims(image_pre, 0)
            
        
            with slim.arg_scope(self.ssd_net.arg_scope(data_format=self.data_format)):
                end_points = {}
                with tf.variable_scope(scope,scope, reuse=self.reuse):
                    # Original VGG-16 blocks.
                    net = slim.repeat(self.inputs, 2, slim.conv2d, int(64*ratio), [3, 3], scope='conv1')
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
        
        self.loss = self.losses(self.batch_size)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        #self.solver = self.optimizer.minimize(self.loss)

#        self.solver = tf.train.MomentumOptimizer(learning_rate = 0.8, momentum=0.9).minimize(self.losses())
            
        self.sess = tf.Session(config=self.config)
        
                    
        if self.ckpt_filename != '':
            
            self.saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(self.ckpt_filename)
            self.saver.restore(self.sess, ckpt)
        else:
            self.sess.run(tf.global_variables_initializer())
            
            with tf.name_scope('model_pruning'):
                self.model_pruning()
#        tf.summary.FileWriter('/home/ubuntu/workspace/fastcnn/log/test', self.sess.graph) 

            
    def test_var(self):
        
        with tf.variable_scope(self.scope) as scope:
            scope.reuse_variables()
            va = tf.get_variable('conv1/conv1_1/weights')
        
        var = self.sess.run(va)
        
        return var
        
    
    def img_preprocessing(self, img):
        image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(self.img_input, None, None, self.net_shape, self.data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        image_4d = tf.expand_dims(image_pre, 0)
        
        pre_img = self.sess.run(image_4d, feed_dict={self.img_input:img})
        
        return pre_img
    
    
    def inference(self,img):

        
        rlogit, self.rpredictions, self.rlocalisations, self.rbbox_img = self.sess.run([self.logits, self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.inputs: img})
        
#        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
#                    self.rpredictions, self.rlocalisations, self.ssd_anchors,
#                    select_threshold=0.5, img_shape=self.net_shape, num_classes=21, decode=True)
#        
#        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
#        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=0.45)
#        
#        rbboxes = np_methods.bboxes_resize(self.rbbox_img, rbboxes)
#        
#
#        return self.rpredictions, self.rlocalisations, self.rbbox_img
        #return rlogit
        
        
        
    def train_op(self):
        
        
         self.loss, _, _ = self.losses()
         #self.loss = self.losses()
        
#         solver = tf.train.MomentumOptimizer(learning_rate = 0.8, momentum=0.9).minimize(self.losses())
#         self.optimizer.compute_gradients(self.losses())
         
#         return solver
         
   
        
    def plot(self,img):

        image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(self.img_input, None, None, self.net_shape, self.data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        image_4d = tf.expand_dims(image_pre, 0)       
        pre_img = self.sess.run(image_4d, feed_dict={self.img_input:img})
        rlogit, self.rpredictions, self.rlocalisations, self.rbbox_img = self.sess.run([self.logits, self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.inputs: pre_img})
        
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        self.rpredictions, self.rlocalisations, self.ssd_anchors,
        select_threshold=0.5, img_shape=self.net_shape, num_classes=21, decode=True)

        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=0.45)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(self.rbbox_img, rbboxes)
        plt_bboxes(img, rclasses, rscores, rbboxes)
    
        

    def flatten_output(self, glabel, glocation, gscore):

        # Flatten out all vectors!
        
        fgclasses = []
        fgscores = []
        fglocalisations = []
        for i in range(len(self.logits)):
            
            fgclasses.append(tf.reshape(glabel[i], [-1]))
            fgscores.append(tf.reshape(gscore[i], [-1]))          
            fglocalisations.append(tf.reshape(glocation[i].astype(np.float32), [-1, 4]))
            
        
        gclasses = tf.concat(fgclasses, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        
        return gclasses, glocalisations, gscores
        # And concat the crap!
        
#        self.gclasses = self.sess.run(tf.concat(fgclasses, axis=0))
#        self.gscores = self.sess.run(tf.concat(fgscores, axis=0))
#        self.glocalisations = self.sess.run(tf.concat(fglocalisations, axis=0))
        
#        return self.gclasses,  self.glocalisations, self.gscores

    def losses(self, batch_size,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope=None):
        
        with tf.name_scope(scope, 'ssd_losses'):
            lshape = tfe.get_shape(self.logits[0], 5)
            num_classes = lshape[-1]
        
            # Flatten out all vectors!
            flogits = []
            flocalisations = []
    
            for i in range(len(self.logits)):
                flogits.append(tf.reshape(self.logits[i], [-1, num_classes]))
                flocalisations.append(tf.reshape(self.localisations[i], [-1, 4]))
    
            logits = tf.concat(flogits, axis=0)     
            localisations = tf.concat(flocalisations, axis=0)
            
#            self.p = tf.Print(logits, [logits])
            
            dtype = logits.dtype
            
    
            # Compute positive matching mask...
            pmask = self.gscore > match_threshold
            fpmask = tf.cast(pmask, dtype)
            n_positives = tf.reduce_sum(fpmask)
    
            # Hard negative mining...
            no_classes = tf.cast(pmask, tf.int32)
            predictions = slim.softmax(logits)
            nmask = tf.logical_and(tf.logical_not(pmask),
                                   self.gscore > -0.5)
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
            
#        with tf.name_scope('cross_entropy_pos'):
#            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
#                                                                  labels=self.glabel)
#            loss_t1 = tf.reduce_sum(loss )           
#            loss1 = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')           
#            #tf.losses.add_loss(loss1)
#            
#        with tf.name_scope('cross_entropy_neg'):
#            loss_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
#                                                                  labels=no_classes)
#            loss2 = tf.div(tf.reduce_sum(loss_n * fnmask), batch_size, name='value')
#            #tf.losses.add_loss(loss2)
#    
#        # Add localization loss: smooth L1, L2, ...
#        with tf.name_scope('localization'):
#            # Weights Tensor: positive mask + random negative.
#            weights = tf.expand_dims(alpha * fpmask, axis=-1)
#            lossl = custom_layers.abs_smooth(localisations - self.glocation)
#            loss3 = tf.div(tf.reduce_sum(lossl * weights), batch_size, name='value')
#            tf.losses.add_loss(loss3)
#            
#        loss = tf.add(loss1, loss2)
#        loss = tf.add(loss, loss3)
#        
        return fnmask
        
                
def plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    
    """
    classes_label =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
     
    
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = str(classes_label[cls_id])
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()        
        
        
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
            




                        
        
        
        
        
        
        