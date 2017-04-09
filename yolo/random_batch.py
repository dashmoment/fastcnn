import numpy as np
from random import shuffle
import os
from scipy import misc
import cv2

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

def yolo_image_random_batch(dirname, batchsize, imagesize, array_type):
    
    cwd = os.getcwd()
    os.chdir(dirname)
    
    files = [x for x in os.listdir(dirname)]
    idx = [x for x in range(len(files))]
    shuffle(idx)
    
    batch = [files[b] for b in idx[0:batchsize]]
    batch_images = []

    for fname in batch:
        img = cv2.imread(fname)
        img_resized = cv2.resize(img, (imagesize[0],imagesize[1]))
        img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray( img_RGB )
        inputs = (img_resized_np/255.0)*2.0-1.0

        batch_images.append(np.array(inputs))
    
    batch_images = np.stack(batch_images)
    
    os.chdir(cwd)
    
    return batch_images


def yolo_image_random_batch_corp(dirname, batchsize, imagesize, array_type):
    
    cwd = os.getcwd()
    os.chdir(dirname)
    
    files = [x for x in os.listdir(dirname)]
    idx = [x for x in range(len(files))]
    shuffle(idx)
    
    batch = [files[b] for b in idx[0:batchsize]]
    batch_images = []

    for fname in batch:
        
        img = cv2.imread(fname)
        
        img_RGB = imcv2_recolor(img)
        img_resized = random_transform(img_RGB,imagesize)
        
        
        cv2.imshow("img_RGB", img_resized)
        cv2.waitKey(5)
        
        #img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray( img_resized )
        inputs = (img_resized_np/255.0)*2.0-1.0

        batch_images.append(np.array(inputs))
    
    batch_images = np.stack(batch_images)
    
    os.chdir(cwd)
    
    return batch_images

def imcv2_recolor(img, a = .1):
    
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    t = [np.random.uniform()]
    t += [np.random.uniform()]
    t += [np.random.uniform()]
    t = np.array(t) * 2. - 1.
    
	# random amplify each channel
    hsv = hsv * (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform() * 2 - 1
    print(up)
    hsv = np.power(hsv/mx, 1. + up * .5)
    rgb = np.array(hsv * 255., np.uint8)
    rgb = cv2.cvtColor(rgb,cv2.COLOR_HSV2RGB)
    
    return rgb

def random_transform(img,imagesize):
    
    img = cv2.resize(img, (imagesize[0],imagesize[1]))
    
    h, w, c = img.shape
    scale = np.random.uniform(low=1.0, high=1.5)
    max_offsetX = (scale -1)*w
    max_offsetY = (scale -1)*h
    offx = int(np.random.uniform() * max_offsetX)
    offy = int(np.random.uniform() * max_offsetY)
    
    rimg = cv2.resize(img, (0,0), fx = scale, fy = scale)
    res = rimg[offy : (offy + h), offx : (offx + w)]
    
    flip = np.random.binomial(1, .5)
    print(flip)
    if flip: res = cv2.flip(res, 1)
    
    
    return res

dname = "/home/dashmoment/workspace/dataset/VOCdevkit/VOC2012/JPEGImages"
b = yolo_image_random_batch_corp(dname, 1, (448,448,3), np.float32)
a = imcv2_recolor(b)


#x = tf.placeholder(tf.float32,(None,255,255,3))
#image_submean = tf.subtract(x, tf.reduce_mean(x))
#
#rconv1_w = tf.Variable(tf.random_normal([11,11,3,48]))
#rconv1_b = tf.Variable(tf.random_normal([48]))
#rconv1_in = tf.nn.conv2d(image_submean, rconv1_w, strides=[1,2,2,1], padding='SAME')
#rconv_in = tf.nn.bias_add(rconv1_in, rconv1_b)
#rconv1_in = tf.nn.relu(rconv_in)
#
#with tf.Session() as sess:
#    init = tf.global_variables_initializer()
#    sess.run(init)  
#    
#    p = sess.run(rconv1_in, feed_dict={x:b}