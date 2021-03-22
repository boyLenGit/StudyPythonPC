from abc import ABC

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential
import numpy
import segyio
import matplotlib.pyplot as plt
import random
import APIFu.LenSegy as LenSegy

conv_kernel_size_mnist, conv_kernel_size_ori = [28, 18, 10, 18, 28], [130, 65, 20, 65, 130]
conv_kernel_size = conv_kernel_size_ori


class Conv2DAe(keras.Model, ABC):
    def __init__(self):
        super(Conv2DAe, self).__init__()
        self.sq1 = Sequential([
            layers.BatchNormalization(), layers.Conv2D(filters=32, kernel_size=[8, 8], activation='relu', padding='same'), layers.MaxPool2D(),
            layers.BatchNormalization(), layers.Conv2D(filters=64, kernel_size=[8, 8], activation='relu', padding='same'), layers.MaxPool2D(),
            layers.BatchNormalization(), layers.Conv2D(filters=64, kernel_size=[4, 4], activation='relu', padding='same'),
            layers.Dense(32*32, activation='relu'),
            layers.Dense(16*16, activation='relu'),
            layers.Dense(32*32, activation='relu'),
            layers.BatchNormalization(), layers.Conv2D(filters=64, kernel_size=[4, 4], activation='relu', padding='same'), layers.UpSampling2D(),
            layers.BatchNormalization(), layers.Conv2D(filters=64, kernel_size=[8, 8], activation='relu', padding='same'), layers.UpSampling2D(),
            layers.BatchNormalization(), layers.Conv2D(filters=1, kernel_size=[8, 8], activation='tanh', padding='same'),
        ])

    def call(self, inputs, training=True, mask=None):
        out1 = self.sq1(inputs)
        self.sq1.summary()
        return out1

def train():
    model1 = Conv2DAe()
    model1.build(input_shape=(None, 256, 256, 1))
