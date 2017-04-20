import tensorflow as tf
import numpy as np
import yolo_netfactory as nf
import random_batch as rb
import YOLO_tiny_tf
import cv2
import voc_utils as voc
import os
import utility as ut
from bs4 import BeautifulSoup as soup
import model_utility as mut
import time


img_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/JPEGImages'
labelfiles = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/ImageSets/Main'
graph_model = '../../model/yolo_ds/fcann_v1.ckpt-4000.meta'
checkpoint_dir = '../../model/yolo_ds'
classes = voc.list_image_sets()
val_list = voc.imgs_from_category_as_list('', 'val', labelfiles)

yolo_old = YOLO_tiny_tf.YOLO_TF()

#Vanilla YOLO_tiny Weight
tmodel_var_list = [
            ['conv1w',[3,3,3,16]],
            ['conv1b',[16]],
            ['conv2w',[3,3,16,32]],
            ['conv2b',[32]],
            ['conv3w',[3,3,32,64]],
            ['conv3b',[64]],
            ['conv4w',[3,3,64,128]],
            ['conv4b',[128]],
            ['conv5w',[3,3,128,256]],
            ['conv5b',[256]],
            ['conv6w',[3,3,256,512]],
            ['conv6b',[512]],
            ['conv7w',[3,3,512,1024]],
            ['conv7b',[1024]],
            ['conv8w',[3,3,1024,1024]],
            ['conv8b',[1024]],
            ['conv9w',[3,3,1024,1024]],
            ['conv9b',[1024]],
            ['fc10w',[50176,256]],
            ['fc10b',[256]],
            ['fc11w',[256,4096]],
            ['fc11b',[4096]],
            ['fc12w',[4096,1470]],
            ['fc12b',[1470]]]

#tmodel_var_list = [
#            ['conv1w',[3,3,3,16]],
#            ['conv1b',[16]],
#            ['conv2w',[3,3,16,16]],
#            ['conv2b',[16]],
#            ['conv3w',[3,3,16,32]],
#            ['conv3b',[32]],
#            ['conv4w',[3,3,32,64]],
#            ['conv4b',[64]],
#            ['conv5w',[3,3,64,128]],
#            ['conv5b',[128]],
#            ['conv6w',[3,3,128,256]],
#            ['conv6b',[256]],
#            ['conv7w',[3,3,256,512]],
#            ['conv7b',[512]],
#            ['conv8w',[3,3,512,512]],
#            ['conv8b',[512]],
#            ['conv9w',[3,3,512,512]],
#            ['conv9b',[512]],
#            ['fc10w',[25088,128]],
#            ['fc10b',[128]],
#            ['fc11w',[128,2048]],
#            ['fc11b',[2048]],
#            ['fc12w',[2048,1470]],
#            ['fc12b',[1470]]]

x = tf.placeholder(tf.float32,(None,448,448,3))
keep_prob = tf.placeholder(tf.float32)
ds_yolo = mut.create_var_tnorm('train',tmodel_var_list)
yolo_ds = nf.yolo_ds("yolo_test",x,ds_yolo,keep_prob)

#Test LSGAN trained model
#ds_yolo = mut.create_var_xavier('train',tmodel_var_list)
#yolo_ds = nf.yolo_ds_all("yolo_train",x,ds_yolo,keep_prob, False)

#file_handler = open('test.txt', 'w')

val_name = val_list[23]

resaver = tf.train.Saver()

tp = 0
fp = 0

tp_old = 0
fp_old = 0

num = 0
num_old = 0
idx = 1
elapse = 0
elapse_old = 0

with tf.Session() as sess2:
    
    
    resaver.restore(sess2, tf.train.latest_checkpoint(checkpoint_dir))
#    c2 = sess2.run(ds_yolo["conv2w"])
    
    for val_name in val_list:
        
        fpath = os.path.join(img_root,val_name+'.jpg')
        w,h,inputs = ut.vocimg_preprocess(fpath)
        src = cv2.imread(fpath)
          
        num = num + ut.calc_objec_num(val_name)
        print("Test File:{}/{}".format(idx,len(val_list)))
        idx = idx + 1
        
        
       
        s = time.clock()
        prob_label_old = yolo_old.sess.run(yolo_old.fc_19, feed_dict={yolo_old.x:inputs})
        e = time.clock()
        elapse_old = elapse_old + e - s
        
        results_old = ut.interpret_output(prob_label_old[0],w,h)

        for i in range(len(results_old)):
            
            tbb = ut.cov_yoloBB2VOC(results_old[i])
            res_old = ut.eval_by_obj(val_name, tbb, 0.5)
            
            if(res_old == 1): tp_old = tp_old +1
            if(res_old == -1): fp_old = fp_old +1
        

        
        s = time.clock()
        prob_label = sess2.run(yolo_ds,feed_dict={x:inputs, keep_prob:0.5})
        e = time.clock()
        elapse = elapse + e-s
        #print("Elapse:{}".format(res))
        
        
        results = ut.interpret_output(prob_label[0],w,h)
        for i in range(len(results)):
            
            tbb = ut.cov_yoloBB2VOC(results[i])
            res = ut.eval_by_obj(val_name, tbb, 0.5)
            
            if(res == 1): tp = tp +1
            if(res == -1): fp = fp +1
        

        
       
    print("Old Avg Elapse:{}".format(elapse_old/idx)) 
    print("Avg Elapse:{}".format(elapse/idx)) 
    print("Old Accuracy:{}".format(tp_old/num))
    print("New Accuracy:{}".format(tp/num)) 
        
#        {} {} {} {} {}\n".format(fname, results[i][5],xmin,ymin,xmax,ymax))
    
    
    #ut.show_results(src,show_res)
    
#    for tmp in bb:
#        cv2.rectangle(src,(int(tmp[0]),int(tmp[1])),(int(tmp[2]),int(tmp[3])),(0,255,0),2)
#    cv2.imshow("gt",src)
#    cv2.waitKey(100)
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        