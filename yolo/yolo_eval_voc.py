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


img_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/JPEGImages'
labelfiles = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/ImageSets/Main'
graph_model = '../../model/test/fcann_v1.ckpt-4000.meta'
checkpoint_dir = '../../model/test'
classes = voc.list_image_sets()
val_list = voc.imgs_from_category_as_list('', 'val', labelfiles)

yolo_old = YOLO_tiny_tf.YOLO_TF()

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

#x = tf.placeholder(tf.float32,(None,448,448,3))
#keep_prob = tf.placeholder(tf.float32)
#ds_yolo = mut.create_var_tnorm('test',tmodel_var_list)
#yolo_ds = nf.yolo_ds_train("yolo_test",x,ds_yolo,keep_prob)

#file_handler = open('test.txt', 'w')

val_name = val_list[18]

fpath = os.path.join(img_root,val_name+'.jpg')
#fpath = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012/JPEGImages/2007_000061.jpg'
w,h,inputs = ut.vocimg_preprocess(fpath)
src = cv2.imread(fpath)
with tf.Session() as sess2:
    
    sess2.run(tf.global_variables_initializer())  
    resaver = tf.train.import_meta_graph(graph_model)
    resaver.restore(sess2, tf.train.latest_checkpoint(checkpoint_dir))
    #ass = sess2.run(tf.get_default_graph().get_tensor_by_name('train/conv1w_1:0'))
    #sess2.run(tf.assign(ds_yolo["conv1w"],ass))
#    for var in ds_yolo:
#        sess2.run(ds_yolo[var].assign(tf.get_collection(var)[0]))
        

#    idx = 1
#    #for fname in val_list:
#         
#    print("Test File:{}/{}".format(idx,len(val_list)))
#    idx = idx + 1
#    prob_label = sess2.run(yolo_ds,feed_dict={x:inputs, keep_prob:0.5})
#    results = ut.interpret_output(prob_label[0],w,h)
#    print(results)
#    
#    prob_label_old = yolo_old.sess.run(yolo_old.fc_19, feed_dict={yolo_old.x:inputs})
#    results_old = ut.interpret_output(prob_label_old[0],w,h)
#    print(results_old)
#    
    #c1 = sess2.run(ds_yolo["conv1w"])
    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    
#    show_res = results_old
#    #ut.show_results(src,results)
#    
#    for i in range(len(show_res)):
#        
#        tbb = ut.cov_yoloBB2VOC(show_res[i])
#        res,bb = ut.eval_by_obj(val_name, tbb, 0.5)
#        print("res:{}".format(res))
#        
##        #file_handler.write("{} {} {} {} {} {}\n".format(fname, results[i][5],xmin,ymin,xmax,ymax))
#    
#    
#    ut.show_results(src,show_res)
#    
#    for tmp in bb:
#        cv2.rectangle(src,(int(tmp[0]),int(tmp[1])),(int(tmp[2]),int(tmp[3])),(0,255,0),2)
#    cv2.imshow("gt",src)
#    cv2.waitKey(100)
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        