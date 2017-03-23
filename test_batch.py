import tensorflow as tf
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from scipy import misc
from numpy import *
import purgeinvalid_img as pi
import datetime


def createmodelname():
    
    filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return filename



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


filename = "../model/fcann_v1.ckpt"

pi.purgeinvalidandRGB_img("../ilsvrc11") #Cleanup Invalid file in directory

sample_batch = randombatch()


with tf.Session() as sess:
    
    
    init = tf.global_variables_initializer()
    sess.run(init)
      
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord) 
    
    try:
        #while not coord.should_stop():
        for i in range(5000000):
                      
            image_tensor = sess.run(sample_batch)
            
            print("Training Step: {}".format(i))
            print(image_tensor[0][0][0][0])
                
        
        sess.close() 
        
           
           
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)
        
        
   








