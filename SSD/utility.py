import tensorflow as tf

def show_all_variable(scope='test'):
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope):
        print (i.name)