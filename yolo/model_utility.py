import tensorflow as tf
import numpy as np

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def create_var_xavier(scope, var_list):

    var_dict = {}
    with tf.name_scope(scope):
        for n in var_list:
            var_dict[n[0]] = tf.get_variable(n[0], shape=n[1], initializer=tf.contrib.layers.xavier_initializer()) 
            

    return var_dict

def create_var_tnorm(scope, var_list, mean = 0.0, stddev = 0.01):

    var_dict = {}
    with tf.name_scope(scope):
        for n in var_list:
           
            var_dict[n[0]] = tf.Variable(tf.truncated_normal(n[1], mean=mean, stddev=stddev),name =n[0])

    return var_dict