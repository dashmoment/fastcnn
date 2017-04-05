import tensorflow as tf
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from scipy import misc
#import net_factory
import time

alpha = 0.1
batch_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train/*.jpg"
test_file = "/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_val/*.jpg"
filename = "../model/voc2012_reg/fcann_v1.ckpt"
logfile = '../log/voc2012_reg'
graph_model = '../model/voc2012_reg/fcann_v1.ckpt-489000.meta'
checkpoint_dir = '../model/voc2012_reg'

def randombatch(batchfile):
    filenames = tf.train.match_filenames_once(batchfile)
    file_queue =  tf.train.string_input_producer( filenames,  shuffle=True)
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(file_queue)
    image_orig = tf.image.decode_jpeg(image_file)
    image = tf.image.resize_images(image_orig, [448, 448])
    image.set_shape((448, 448, 3))   
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


ds_yolo = {
        'conv1w':tf.Variable(tf.truncated_normal([3,3,3,16], stddev=0.1)),
        'conv1b':tf.Variable(tf.truncated_normal([16], stddev=0.1)),
        'conv2w':tf.Variable(tf.truncated_normal([3,3,16,32], stddev=0.1)),
        'conv2b':tf.Variable(tf.truncated_normal([32], stddev=0.1)),
        'conv3w':tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.1)),
        'conv3b':tf.Variable(tf.truncated_normal([64], stddev=0.1)),
        'conv4w':tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1)),
        'conv4b':tf.Variable(tf.truncated_normal([128], stddev=0.1)),
        'conv5w':tf.Variable(tf.truncated_normal([3,3,128,256], stddev=0.1)),
        'conv5b':tf.Variable(tf.truncated_normal([256], stddev=0.1)),
        'conv6w':tf.Variable(tf.truncated_normal([3,3,256,512], stddev=0.1)),
        'conv6b':tf.Variable(tf.truncated_normal([512], stddev=0.1)),
        'conv7w':tf.Variable(tf.truncated_normal([3,3,512,1024], stddev=0.1)),
        'conv7b':tf.Variable(tf.truncated_normal([1024], stddev=0.1)),
        'conv8w':tf.Variable(tf.truncated_normal([3,3,1024,1024], stddev=0.1)),
        'conv8b':tf.Variable(tf.truncated_normal([1024], stddev=0.1)),
        'conv9w':tf.Variable(tf.truncated_normal([3,3,1024,1024], stddev=0.1)),
        'conv9b':tf.Variable(tf.truncated_normal([1024], stddev=0.1)),
        'fc10w':tf.Variable(tf.truncated_normal([50176,256], stddev=0.1)),
        'fc10b':tf.Variable(tf.truncated_normal([256], stddev=0.1)),
        'fc11w':tf.Variable(tf.truncated_normal([256,4096], stddev=0.1)),
        'fc11b':tf.Variable(tf.truncated_normal([4096], stddev=0.1)),
        'fc12w':tf.Variable(tf.truncated_normal([4096,1470], stddev=0.1)),
        'fc12b':tf.Variable(tf.truncated_normal([1470], stddev=0.1))
        }

x = tf.placeholder(tf.float32,(None,448,448,3))
sample_batch = randombatch(batch_file)

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
    
    
              
with tf.name_scope("ds_yolo"):
    
    with tf.name_scope("conv1"):
        conv1 = conv(x, ds_yolo['conv1w'], ds_yolo['conv1b'],2)
        
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
        
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())  
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)  
    
    try:
        
        img = sess.run(sample_batch)
        c1 = sess.run(fc12, feed_dict={x:img})
        
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    