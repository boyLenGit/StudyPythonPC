from abc import ABC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential
import numpy
import segyio
import matplotlib.pyplot as plt
import random
import APIFu.LenSegy as LenSegy
import APIFu.LenDataProcess as LenDataProcess
"""初始化变量"""
dimension2d_input = 256
path_segy_file = 'E:/Research/data/F3_entire.segy'


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
    data = LenSegy.len_read_segy_file(path_segy_file)['trace_raw'][:1000]
    print('$ ', 5000//256)
    data_been_kernel, dimension_kernels = LenDataProcess.len_kernel_2d_cut(data)
    data1 = LenDataProcess.len_kernel_2d_merge(data_been_kernel, data, dimension_kernels)
    # model1 = Conv2DAe()
    # model1.build(input_shape=(None, dimension2d_input, dimension2d_input, 1))  # [None, 256, 256, 1]
    # print(type(data1))
    LenSegy.len_save_segy_multi_process(data1, path_segy_file, 'F:/test_LenKernel2D.segy')


if __name__ == '__main__':
    train()
