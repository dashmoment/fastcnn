import tensorflow as tf
import numpy as np
import yolo_netfactory as nf
import random_batch as rb
import YOLO_tiny_tf
import cv2
from voc_parser import voc_utils

import time

def iou(box1,box2):
        tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

def interpret_output(intputs):
    
        threshold = 0.05
        iou_threshold = 0.2
        classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
        output =  np.copy(intputs)
    
        w_img = 448
        h_img = 448
        
        probs = np.zeros((7,7,2,20))
        class_probs = np.reshape(output[0:980],(7,7,20))
        scales = np.reshape(output[980:1078],(7,7,2))
        boxes = np.reshape(output[1078:],(7,7,2,4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

        boxes[:,:,:,0] += offset
        boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
        boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
        boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
        boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])
        
        boxes[:,:,:,0] *= w_img
        boxes[:,:,:,1] *= h_img
        boxes[:,:,:,2] *= w_img
        boxes[:,:,:,3] *= h_img

        for i in range(2):
            for j in range(20):
                probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

        filter_mat_probs = np.array(probs>=threshold,dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]
        print(classes_num_filtered)
        
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0 : continue
            for j in range(i+1,len(boxes_filtered)):
                if iou(boxes_filtered[i],boxes_filtered[j]) > iou_threshold : 
                    probs_filtered[j] = 0.0
        
        filter_iou = np.array(probs_filtered>0.0,dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

        return result

def init_yolo_weight(sess,yolo_cls, ds_yolo):
    
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
    
    for var in key_pairs:
        yolo_var = yolo.sess.run(tf.get_default_graph().get_tensor_by_name(key_pairs[var]+':0'))
        op = tf.assign(ds_yolo[var], yolo_var)
        sess.run(op)



#Vanilla Yolo
yolo = YOLO_tiny_tf.YOLO_TF()

ds_yolo = {
        'conv1w':tf.Variable(tf.truncated_normal([3,3,3,16], mean=0, stddev=0.01)),
        'conv1b':tf.Variable(tf.truncated_normal([16], mean=0, stddev=0.01)),
        'conv2w':tf.Variable(tf.truncated_normal([3,3,16,32], mean=0,stddev=0.01)),
        'conv2b':tf.Variable(tf.truncated_normal([32], mean=0, stddev=0.01)),
        'conv3w':tf.Variable(tf.truncated_normal([3,3,32,64], mean=0, stddev=0.01)),
        'conv3b':tf.Variable(tf.truncated_normal([64], mean=0, stddev=0.01)),
        'conv4w':tf.Variable(tf.truncated_normal([3,3,64,128], mean=0, stddev=0.01)),
        'conv4b':tf.Variable(tf.truncated_normal([128], mean=0, stddev=0.01)),
        'conv5w':tf.Variable(tf.truncated_normal([3,3,128,256], mean=0, stddev=0.01)),
        'conv5b':tf.Variable(tf.truncated_normal([256], mean=0, stddev=0.01)),
        'conv6w':tf.Variable(tf.truncated_normal([3,3,256,512], mean=0, stddev=0.01)),
        'conv6b':tf.Variable(tf.truncated_normal([512], mean=0, stddev=0.01)),
        'conv7w':tf.Variable(tf.truncated_normal([3,3,512,1024], mean=0, stddev=0.01)),
        'conv7b':tf.Variable(tf.truncated_normal([1024], mean=0, stddev=0.01)),
        'conv8w':tf.Variable(tf.truncated_normal([3,3,1024,1024], mean=0, stddev=0.01)),
        'conv8b':tf.Variable(tf.truncated_normal([1024], mean=0, stddev=0.01)),
        'conv9w':tf.Variable(tf.truncated_normal([3,3,1024,1024], mean=0, stddev=0.01)),
        'conv9b':tf.Variable(tf.truncated_normal([1024], mean=0, stddev=0.01)),
        'fc10w':tf.Variable(tf.truncated_normal([50176,256], mean=0, stddev=0.01)),
        'fc10b':tf.Variable(tf.truncated_normal([256], mean=0, stddev=0.01)),
        'fc11w':tf.Variable(tf.truncated_normal([256,4096], mean=0, stddev=0.01)),
        'fc11b':tf.Variable(tf.truncated_normal([4096], mean=0, stddev=0.01)),
        'fc12w':tf.Variable(tf.truncated_normal([4096,1470], mean=0, stddev=0.01)),
        'fc12b':tf.Variable(tf.truncated_normal([1470], mean=0, stddev=0.01))
        }


batch_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train"
test_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_val"
filename = "../../model/yolo_ds/fcann_v1.ckpt"
logfile = '../../log/yolo_ds'
graph_model = '../../model/yolo_ds/fcann_v1.ckpt-62000.meta'
checkpoint_dir = '../../model/yolo_ds'


batch_size = 1

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32,(None,448,448,3))
label = tf.placeholder(tf.float32,(None,1470))


yolo_ds = nf.yolo_ds(x,ds_yolo,keep_prob)
res_value = tf.subtract(yolo_ds, label)
loss = tf.sqrt(tf.reduce_sum(tf.square(res_value)))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
#tf.summary.scalar("train_RMSE",loss, collections=['train'])

fromfile = "test/person.jpg"
img = cv2.imread(fromfile)
img_resized = cv2.resize(img, (448, 448))
img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
img_resized_np = np.asarray( img_RGB )
inputs = np.zeros((1,448,448,3),dtype='float32')
inputs[0] = (img_resized_np/255.0)*2.0-1.0

with tf.Session() as sess2:
    
    
    saver = tf.train.Saver()    
    sess2.run(tf.global_variables_initializer())  
        
    resaver = tf.train.import_meta_graph(graph_model)
    resaver.restore(sess2, tf.train.latest_checkpoint(checkpoint_dir))
    
    for var in ds_yolo:
        sess2.run(ds_yolo[var].assign(tf.get_collection(var)[0]))
    a = sess2.run(ds_yolo[var])
    #image_src =  rb.yolo_image_random_batch(batch_file, batch_size, (448,448,3), np.float32)
    
    s = time.clock()
    prob_label = yolo.sess.run(yolo.fc_19, feed_dict={yolo.x:inputs})
    e = time.clock()
    print("Origin Elapse:{}".format(e-s))
    
    s = time.clock()
    prob_label_new = sess2.run(yolo_ds,feed_dict={x:inputs, keep_prob:0.5})
    e = time.clock()
    print("DS Elapse:{}".format(e-s))
    
    
    loss = sess2.run(loss,feed_dict={x:inputs, label:prob_label,keep_prob:0.5})
       
                      
    results = interpret_output(prob_label[0])
#    yolo.show_results(img_resized,results)
    
    
    results_new = interpret_output(prob_label_new[0])
#    yolo.show_results(img_resized,results_new)
    
    
    
   
    sess2.close()  
    


 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    