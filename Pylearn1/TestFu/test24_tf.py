#
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential
import matplotlib.pyplot as plt


def func1():
    tf1 = [[1, 2, 3], [4, 5, 6]]
    tf1_mean = tf.reduce_mean(tf.cast(tf1, tf.float32))
    print(tf1_mean)

    tf2 = tf.constant([[1., 2., 3.], [4., 5., 6.]])
    tf2_mean = tf.reduce_mean(tf2)


if __name__ == '__main__':
    func1()
