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
    batch_size = 1
    
    images = tf.train.shuffle_batch(
            [image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + (num_preprocess_threads +50)*batch_size,
            min_after_dequeue=min_queue_examples)
    
    return images

#output = vanilla_alex(x)
train_x = np.zeros((1, 227,227,3)).astype(np.float32)
train_y = np.zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]


x = tf.placeholder(tf.float32, (None,) + xdim)

sample_batch = randombatch()
 
with tf.Session() as sess:
    
    new_saver = tf.train.import_meta_graph('../model/half_2_1e-4_dstride_p2/fcann_v1.ckpt-13000.meta')
    #new_saver.restore(sess, tf.train.latest_checkpoint('../model'))
    new_saver.restore(sess, '../model/half_2_1e-4_dstride_p2/fcann_v1.ckpt-13000')
    
    tw1 = tf.Variable(sess.run(tf.get_collection("w1")[0]))
    tb1 = tf.Variable(sess.run(tf.get_collection("b1")[0]))
    
    
    net = net_factory.vanilla_alex_full(x)
    mininet = net_factory.mini_alex_full(x,tw1,tb1)
    
    init = tf.global_variables_initializer()
    sess.run(init)  
    
        
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord) 
    
    try:
        
        image_tensor = sess.run(sample_batch)
        w1 = sess.run(tw1)
        output = sess.run(net, feed_dict = {x:image_tensor})
        output2 = sess.run(mininet, feed_dict = {x:image_tensor})
        
        plt.figure(1)
        plt.subplot(211)
        plt.plot(output[0])
        plt.figure(2)
        plt.plot(output2[0])
        
        for input_im_ind in range(output.shape[0]):
            
            inds = np.argsort(output)[input_im_ind,:]
            inds2 = np.argsort(output2)[input_im_ind,:]
            
                
            for j in range(0,len(inds)):
                if j < len(inds)-5 : 
                    output[input_im_ind, inds[j]] = 0
                    output2[input_im_ind, inds2[j]] = 0
                if j >= len(inds)-5:
                    output2[input_im_ind, inds2[j]] = 1
                    output[input_im_ind, inds[j]] = 1
        print("Origin:{}, Mini:{}".format(class_names[inds[-1]],class_names[inds2[-1]]))
        plt.figure(3)
        plt.subplot(212)
        plt.plot(output[0])
        plt.plot(output2[0])
        cost = tf.reduce_mean(-tf.reduce_sum(tf.multiply(output,tf.log(output2)),1))     
        loss = sess.run(cost)
        
        
        print("Avg Loss:{}".format(loss))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)
    sess.close()