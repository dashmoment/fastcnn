import tensorflow as tf
import numpy as np
import cv2
import yolo_netfactory as nf
import random_batch as rb
import YOLO_tiny_tf
import voc_utils as voc
import os
import utility as ut
from bs4 import BeautifulSoup as soup
import model_utility as mut
import time
import model_utility as mu
import matplotlib.pyplot as plt

key_pairs = {
            'conv1w':'Variable',
            'conv1b':'Variable_1',
            'conv2w':'Variable_2',
            'conv2b':'Variable_3',
            'conv3w':'Variable_4',
            'conv3b':'Variable_5',
            'conv4w':'Variable_6',
            'conv4b':'Variable_7',
            'conv5w':'Variable_8',
            'conv5b':'Variable_9',
            'conv6w':'Variable_10',
            'conv6b':'Variable_11',
            'conv7w':'Variable_12',
            'conv7b':'Variable_13',
            'conv8w':'Variable_14',
            'conv8b':'Variable_15',
            'conv9w':'Variable_16',
            'conv9b':'Variable_17',
            'fc10w':'Variable_18',
            'fc10b':'Variable_19',
            'fc11w':'Variable_20',
            'fc11b':'Variable_21',
            'fc12w':'Variable_22',
            'fc12b':'Variable_23',
            
            }



def recursive_create_var(scope, Nlayers, reduce_percent, init_layers):
    
    scope_dict = {}
    
    for idx in range(Nlayers):
        
        scope_name = scope + '_' + str(idx)
        with tf.variable_scope(scope_name):
            
            name_dict = []
            
            for i in range(len(init_layers)):

                shape = init_layers[i][1]
                
                if idx >= 1: 
                                       
                    if i > 0 and len(shape)  > 1 : shape[-2] = int(init_layers[i-1][1][-1])
                    
                    shape[-1] = int(init_layers[i][1][-1]*reduce_percent)
                    
                    
                init_layers[i][1] = shape
                
                if init_layers[i][0] == 'fc12w':  init_layers[i][1][-1] = 1470
                if init_layers[i][0] == 'fc12b':  init_layers[i][1][-1] = 1470          
                if init_layers[i][0] == 'fc10w':
                
                    if idx == 0:
                        init_layers[i][1][-2] = 50176
                    if idx == 1:
                        init_layers[i][1][-2] = 47628
                    if idx == 2:
                        init_layers[i][1][-2] = 32095
                   
          
                tf.get_variable(init_layers[i][0],init_layers[i][1], initializer=tf.contrib.layers.xavier_initializer())

                name_dict.append(init_layers[i][0])
                
        scope_dict[scope_name] = name_dict 

    return scope_dict




img_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2007/Test/JPEGImages'
shrink_ratio = 0.95

if os.path.isdir(img_root):

    labelfiles = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2007/Test/ImageSets/Main'
else:
    img_root = '/home/dashmoment/workspace/dataset/VOCdevkit/VOC2007/Test/JPEGImages'
    labelfiles = '/home/dashmoment/workspace/dataset/VOCdevkit/VOC2007/Test/ImageSets/Main'

yolo_old = YOLO_tiny_tf.YOLO_TF()
var_scope = 'recursive_0'

batch_size = 1

with tf.device('/gpu:1'):
    classes = voc.list_image_sets()
    val_list = voc.imgs_from_category_as_list('', 'test', labelfiles)
    
    keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
    x = tf.placeholder(tf.float32,(None,448,448,3), name='input_batch')
    label = tf.placeholder(tf.float32,(None,1470), name='labels')
    
    tvalid_class = tf.placeholder(tf.float32,(None,7,7,20), name='class')
    tvalid_xy = tf.placeholder(tf.float32,(None,7,7,2,2), name='xy')
    tvalid_wh = tf.placeholder(tf.float32,(None,7,7,2,2), name='wh')
    tlabel_xy = tf.placeholder(tf.float32,(None,7,7,2,2), name='label_xy')
    tlabel_wh = tf.placeholder(tf.float32,(None,7,7,2,2), name='label_wh')
    tconf_weight = tf.placeholder(tf.float32,(None,7,7,2), name='label_wh')
    tlabel_conf = tf.placeholder(tf.float32,(None,7,7,2), name='label_wh')
    
    model_ticket={'root':'yolo_tiny', 'branch':'vanilla'}
    init_layers = mu.model_zoo(model_ticket)
    var_dict = recursive_create_var('recursive', 3, shrink_ratio, init_layers)
    yolo_ds = nf.glosso_train(var_scope, 'test', x, var_dict, keep_prob, False)  
    
    pre_cls = tf.reshape(tf.slice(yolo_ds, [0,0],[-1, 980]), (-1,7,7,20))
    pre_conf = tf.reshape(tf.slice(yolo_ds, [0,980],[-1, 98]), (-1,7,7,2))
    pre_offset = tf.reshape(tf.slice(yolo_ds, [0,1078],[-1, 392]), (-1,7,7,2,4))
    
    pre_xy, pre_wh = tf.split(pre_offset,2,axis=4)
    
    class_loss = tf.reduce_sum(tf.square(tf.subtract(tf.multiply(pre_cls, tvalid_class), tvalid_class)))
    
    box_xy_loss = tf.reduce_sum(tf.square(tf.subtract(tf.multiply(pre_xy, tvalid_xy),tlabel_xy)))
    pre_wh_sqrt = tf.sqrt(tf.multiply(pre_wh, tvalid_wh))
    label_wh_sqrt = tf.sqrt(tlabel_wh)
    box_wh_loss = tf.reduce_sum(tf.square(tf.subtract(pre_wh_sqrt,label_wh_sqrt)))
    tbias_loss = 5*(box_xy_loss + box_wh_loss)
    
    tconf_loss =  tf.reduce_sum(tf.multiply(tf.square(tf.subtract(tlabel_conf, pre_conf)), tconf_weight))
    
    tloss = (class_loss + tbias_loss + tconf_loss)/batch_size

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config = config) as sess:
    
     sess.run(tf.global_variables_initializer())  
     mu.weight_pruning(yolo_old, sess, var_dict, var_scope)
     
     tdata_labels = mu.gnerate_dl_pairs(yolo_old,img_root, batch_size, (448,448,3))
     inputs = tdata_labels['data']
     prob_label_old = tdata_labels['label']
     

    
     

#     fpath = os.path.join(img_root, val_list[0]+'.jpg')   
#     w,h,inputs = ut.vocimg_preprocess(fpath)
#     src = cv2.imread(fpath)
#     
#     fpath2 = os.path.join(img_root, val_list[1]+'.jpg') 
#     w2,h2,inputs2 = ut.vocimg_preprocess(fpath2)
#     src2 = cv2.imread(fpath2)
#     
#     batch = [inputs[0], inputs[0],inputs2[0]]
#     inputs = np.stack(batch)
     

     
     s = time.clock()
     prob_label_old = yolo_old.sess.run(yolo_old.fc_19, feed_dict={yolo_old.x:inputs})
     e = time.clock()    
     print("Time old:{}".format(e-s))
     
     s = time.clock()     
     prob_label = sess.run(yolo_ds,feed_dict={x:inputs, keep_prob:1})
     e = time.clock()
     print("Time:{}".format(e-s))
     
#     results_old = ut.interpret_output(prob_label_old[0],w,h)
#     results = ut.interpret_output(prob_label[0],w,h)   
       
     #-------Label preprocess----------
     
     prob_threshold = 0.1
     
     
     rlabel_boxes = np.zeros((7,7,2,4))
     label_probs = np.reshape(prob_label_old[:,0:980],(-1,7,7,20))
     label_scales = np.reshape(prob_label_old[:,980:1078],(-1,7,7,2))
     label_boxes = np.reshape(prob_label_old[:,1078:],(-1,7,7,2,4))
     conf = np.zeros((label_probs.shape[0], 7,7,2,20))
     
     for i in range(2):
            for j in range(20):
                conf[:,:,:,i,j] = np.multiply(label_probs[:,:,:,j],label_scales[:,:,:,i])
                filter_mat_probs = np.array(conf>=prob_threshold,dtype='int')  
     
     valid_class = np.multiply(conf, filter_mat_probs)
     valid_class = np.sum(valid_class, axis=-2)
     valid_class = np.array(valid_class>0,dtype='int')  
     
     valid_box = np.multiply(conf, filter_mat_probs)
     valid_box = np.sum(valid_box, axis=-1)
     valid_box = np.array(valid_box>0,dtype='int')     
     
     for i in range(2):
         for j in range(4):
             label_boxes[:,:,:,i,j] = np.multiply(label_boxes[:,:,:,i,j], valid_box[:,:,:,i])
     
     label_xy , label_wh = np.split(label_boxes,2,axis=-1)
     valid_xy = np.array(label_xy>0,dtype='int')  
     valid_wh = np.array(label_wh>0,dtype='int') 
     
     label_conf = np.multiply(valid_box, label_scales)
     w_obj = np.array(np.abs(label_conf)>0,dtype='int')
     w_nobj = np.array(np.abs(label_conf)==0,dtype='int')
     w = w_obj + 0.5*w_nobj
     
     #------tf class loss-----------
     
     class_loss_t = sess.run(class_loss, feed_dict={x:inputs, keep_prob:1, tvalid_class:valid_class})
     
     #--------tf box loss-------------

     xy_loss_t = sess.run(box_xy_loss, feed_dict={x:inputs, keep_prob:1, tvalid_xy:valid_xy, tlabel_xy:label_xy})
     wh_loss_t = sess.run(box_wh_loss, feed_dict={x:inputs, keep_prob:1, tvalid_wh:valid_wh, tlabel_wh:label_wh})     
     bias_loss = sess.run(tbias_loss, feed_dict={x:inputs, keep_prob:1, tvalid_xy:valid_xy, tlabel_xy:label_xy,tvalid_wh:valid_wh, tlabel_wh:label_wh})

    #--------tf conf loss-------
    
     conf_loss = sess.run(tconf_loss, feed_dict={x:inputs, keep_prob:1, tlabel_conf:label_conf, tconf_weight:w})
     
     feeddict = {x:inputs, keep_prob:1, tvalid_class:valid_class,  tvalid_xy:valid_xy, tlabel_xy:label_xy,tvalid_wh:valid_wh, tlabel_wh:label_wh, tlabel_conf:label_conf, tconf_weight:w}
     loss = sess.run(tloss, feed_dict = feeddict)
     
     
#     #-------calculate loss--------------
     
     label = prob_label_old[0]
     predict = prob_label[0]

     conf = np.zeros((7,7,2,20))
     rlabel_boxes = np.zeros((7,7,2,4))
     pre_conf = np.zeros((7,7,2,20))
     label_probs = np.reshape(label[0:980],(7,7,20))
     label_scales = np.reshape(label[980:1078],(7,7,2))
     label_boxes = np.reshape(label[1078:],(7,7,2,4))
     
     
     pre_probs = np.reshape(predict[0:980],(7,7,20))
     pre_scales = np.reshape(predict[980:1078],(7,7,2))
     pre_boxes = np.reshape(predict[1078:],(7,7,2,4))
     
     for i in range(2):
            for j in range(20):
                conf[:,:,i,j] = np.multiply(label_probs[:,:,j],label_scales[:,:,i])
                pre_conf[:,:,i,j] = np.multiply(pre_probs[:,:,j],pre_scales[:,:,i])
                
     filter_mat_probs = np.array(conf>=0.1,dtype='int')    
     conf = np.multiply(conf, filter_mat_probs)
     filter_mat_probs = np.nonzero(filter_mat_probs)
     
     #-------class loss-----------
     
     for k in range(len(filter_mat_probs[0])):
         conf[filter_mat_probs[0][k],filter_mat_probs[1][k],filter_mat_probs[2][k],filter_mat_probs[3][k]] = 1
     
     rconf = np.sum(conf, axis=2)
     rconf = np.array(rconf>=0.1,dtype='int')
        
     rprob = np.multiply(pre_probs, rconf)
     
     class_loss2 = np.sum(np.square(rprob - rconf))
     
     
     #-------box loss-----------
     
     bconf = np.sum(conf, axis=3)
     bconf = np.array(bconf>=0.1,dtype='int')
     bconf = np.reshape(bconf, (7,7,2,1))
     
     for i in range(2):
         
         rlabel_boxes[:,:,i,:] = np. multiply(label_boxes[:,:,i,:], bconf[:,:,i])
         
     rlabel_boxesidx = np.array(np.abs(rlabel_boxes)> 0,dtype='int')
     rpre_boxes = np.multiply(pre_boxes, rlabel_boxesidx)
     
     bias_loss2 = 0
     xy_loss = 0
     wh_loss = 0
     
     for i in range(4):
         
         if i == 0 or i ==1:
             bias_loss2 = bias_loss2 + np.sum(np.square(rpre_boxes[:,:,:,i] - rlabel_boxes[:,:,:,i]))
             xy_loss = xy_loss +  np.sum(np.square(rpre_boxes[:,:,:,i] - rlabel_boxes[:,:,:,i]))
         if i == 2 or i==3:
             bias_loss2 = bias_loss2 + np.sum(np.square(np.sqrt(rpre_boxes[:,:,:,i]) - np.sqrt(rlabel_boxes[:,:,:,i])))
             wh_loss = wh_loss + np.sum(np.square(np.sqrt(rpre_boxes[:,:,:,i]) - np.sqrt(rlabel_boxes[:,:,:,i])))
     bias_loss2 = 5*bias_loss2
         
     #------confidence loss--------
     c_conf = np.reshape(bconf, (7,7,2))
     rlabel_scales = np.multiply(label_scales, c_conf)
     rw_obj = np.array(np.abs(rlabel_scales)>0,dtype='int')
     rw_nobj = np.array(np.abs(rlabel_scales)==0,dtype='int')
     rw = rw_obj + 0.5*rw_nobj
     
     conf_loss2 = np.sum(np.multiply(rw, np.square((rlabel_scales-pre_scales))))
     
     
     
     
     
     
     
     
     
     
         
         
         
         