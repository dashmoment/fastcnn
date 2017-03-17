import tensorflow as tf
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from numpy import *
import purgeinvalid_img as pi
import datetime


def createmodelname():
    
    filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return filename

path  = "./model/fcann_"
ftype = ".ckpt" 
filename = "./model/fcann_v1.ckpt"

pi.purgeinvalidandRGB_img("../ilsvrc11") #Cleanup Invalid file in directory

def randombatch():
    filenames = tf.train.match_filenames_once("../ilsvrc11/*.jpg")
    file_queue =  tf.train.string_input_producer( filenames,  shuffle=True)
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(file_queue)
    image_orig = tf.image.decode_jpeg(image_file)
    image = tf.image.resize_images(image_orig, [227, 227])
    image.set_shape((227, 227, 3))   
    num_preprocess_threads = 5
    min_queue_examples = 500
    batch_size = 128
    
    images = tf.train.shuffle_batch(
            [image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + (num_preprocess_threads +50)*batch_size,
            min_after_dequeue=min_queue_examples)
    
    return images


train_x = np.zeros((1,227,227,3)).astype(float32)
train_y = np.zeros((1,1000))
x_dim = train_x.shape[1:]
y_dim = train_y.shape[1:]

x = tf.placeholder(tf.float32,(None,) + x_dim)


#Load trained model
net_data = np.load("bvlc_alexnet.npy", encoding='latin1').item()

#conv1

k_size = 11
stride = 4
out_size = 96

conv1_w = tf.Variable(net_data["conv1"][0])
conv1_b = tf.Variable(net_data["conv1"][1])
conv1_in = tf.nn.conv2d(x,conv1_w, strides=[1,stride,stride,1], padding='SAME')
conv_in = tf.nn.bias_add(conv1_in, conv1_b)
conv1 = tf.nn.relu(conv_in)

radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
#maxpool1 = tf.nn.max_pool(lrn1, ksize = [1,1,1,2],strides=[1,1,1,2], padding='VALID')

rconv1_w = tf.Variable(tf.random_normal([11,11,3,96]))
rconv1_b = tf.Variable(tf.random_normal([96]))

stride = stride
rconv1_in = tf.nn.conv2d(x, rconv1_w, strides=[1,stride,stride,1], padding='SAME')
rconv_in = tf.nn.bias_add(rconv1_in, rconv1_b)
rconv1_in = tf.nn.relu(rconv_in)
rlrn1 = tf.nn.local_response_normalization(rconv1_in,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
rmaxpool1 = tf.nn.max_pool(rlrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
saver = tf.train.Saver({"w1":rconv1_w, "w2":rconv1_b})


raw = tf.placeholder(tf.float32,(None,) + x_dim)
image_submean = tf.subtract(raw, tf.reduce_mean(raw))

        
y = tf.subtract(maxpool1, rmaxpool1)
res = tf.reduce_mean(tf.multiply(y,y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(res)

sample_batch = randombatch()


with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)
      
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord) 
    
    try:
        #while not coord.should_stop():
        for i in range(151):
            # Run training steps or whatever
            image_tensor = sess.run(sample_batch)
            print(image_tensor[0][0][0][0])
            
            resimage = sess.run(image_submean, feed_dict={raw:image_tensor}) 
            output = sess.run(maxpool1, feed_dict = {x:resimage})
        
            sess.run(train_step, feed_dict = {x:resimage})
    
            print("Training Step: {}".format(i))
            if i%50 == 0:
                loss = sess.run(res, feed_dict = {x:resimage}) 
                print("Res of Step {}:{}".format(i,loss))
                
          
                saved_model = saver.save(sess, filename, global_step=i)
                print("Save Model:{}".format(filename))
                
           
          
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)
        
        









