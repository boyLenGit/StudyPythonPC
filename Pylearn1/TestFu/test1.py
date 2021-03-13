import tensorflow as tf
import numpy

a3 = tf.Variable([[1, 2], [3, 4.1], [3, 4.1]])
a4 = tf.Variable([[11, 22], [33, 44.1], [33, 44.1]])
a5 = tf.concat([a3, a4], axis=0)
print(a3, a4, a5)

def func1():
    return 'test'

class IJO():
    pass
