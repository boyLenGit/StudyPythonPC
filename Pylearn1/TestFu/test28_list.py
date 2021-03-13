import tensorflow as tf
import numpy
a4_1 = [[1, 2, 3, 0, 0],
        [4, 0, 5, 0, 0],
        [6, 7, 8, 0, 0]]
a4_2 = [[1, 2, 3, 0, 0],
        [4, 0, 5, 0, 0],
        [6, 7, 8, 0, 0]]
'''a1 = tf.constant(a4_1)
a2 = tf.constant(a4_2)'''
a1, a2 = numpy.array(a4_1), numpy.array(a4_2)
print((a1+a2)/2)
