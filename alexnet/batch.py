#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:09:02 2017

@author: ubuntu
"""
import tensorflow as tf
import purgeinvalid_img as pi

pi.purgeinvalid_img("./ilsvrc11") #Cleanup Invalid file in directory

 # Produce FIFO Queue, need to call queue Coordinator before run
file_queue =  tf.train.string_input_producer(tf.train.match_filenames_once("./ilsvrc11/*.jpg"))
image_reader = tf.WholeFileReader()
_, image_file = image_reader.read(file_queue)
image_orig = tf.image.decode_jpeg(image_file)
image = tf.image.resize_images(image_orig, [300, 300])
image.set_shape((300, 300, 3))

num_preprocess_threads = 1
min_queue_examples = 200
batch_size = 100

images = tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

x = tf.placeholder(tf.int32, [None, 300,300,3], name="input_x")
y = tf.multiply(x,x)

with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    image_tensor = sess.run(images)
    #print(image_tensor)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
    
    batch = sess.run(x, feed_dict={x:image_tensor})
    