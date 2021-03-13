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
read_num = 200000
to_be_one = 10000.
path_model_weights = 'D:/boyLen/Python/Pylearn1/TensorFlowFu/TensorTeacher/data/005_Conv2DAE_v1.2'
# path_segy_file = 'E:/Research/data/F3_entire.segy'
path_segy_file = 'E:/Research/data/ArtificialData/extract_segy_valid_line_v1.segy'
path_segy_mix_noisy_file = 'E:/Research/data/ArtificialData/test_Mix_Noisy_v2.segy'


class Conv2DAe(keras.Model, ABC):
    def __init__(self):
        super(Conv2DAe, self).__init__()
        self.sq1 = Sequential([
            layers.BatchNormalization(),
            layers.Conv2D(filters=32, kernel_size=[8, 8], activation='relu', padding='same'), layers.MaxPool2D(),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=[8, 8], activation='relu', padding='same'), layers.MaxPool2D(),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=[4, 4], activation='relu', padding='same'),
            layers.Dense(32 * 32, activation='relu'),
            layers.Dense(16 * 16, activation='relu'),
            layers.Dense(32 * 32, activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=[4, 4], activation='relu', padding='same'), layers.UpSampling2D(),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=[8, 8], activation='relu', padding='same'), layers.UpSampling2D(),
            layers.BatchNormalization(),
            layers.Conv2D(filters=1, kernel_size=[8, 8], activation='tanh', padding='same'),
        ])

    def call(self, inputs, training=True, mask=None):
        out1 = self.sq1(inputs)
        # self.sq1.summary()
        return out1


def train(train_time=100):
    data_origin, data_origin_noisy = LenSegy.len_read_segy_file(path_segy_file)['trace_raw'][:read_num], \
                                     LenSegy.len_read_segy_file(path_segy_mix_noisy_file)['trace_raw'][:read_num]
    data_kernel2d_cut, dimension_kernels = LenDataProcess.len_kernel_2d_cut(data_origin)
    data_kernel2d_cut_noisy, dimension_kernels_noisy = LenDataProcess.len_kernel_2d_cut(data_origin_noisy)
    dataset = LenDataProcess.segy_to_tf_for_result_2output(data1=data_kernel2d_cut, data2=data_kernel2d_cut_noisy,
                                                           to_be_one=to_be_one, reshape_dim=None, expand_axis=3,
                                                           batch_size=8)
    # data_kernel2d_merge = LenDataProcess.len_kernel_2d_merge(data_kernel2d_cut, data_origin, dimension_kernels)
    # ------------------------------------- 初始化并训练模型 ---------------------------------------
    model1 = Conv2DAe()
    model1.build(input_shape=(None, dimension2d_input, dimension2d_input, 1))  # [None, 256, 256, 1]
    len_train_time = len(dataset)
    print('总长度：', len_train_time)
    optimizer1 = tf.optimizers.Adam(lr=0.001, name='SGD')
    for epoch1 in range(train_time):
        print('@ 第{0}次训练：       '.format(epoch1 + 1), end='')
        for step, (x_train, x_train_noisy) in enumerate(dataset):
            with tf.GradientTape() as tape:
                x_result = model1(x_train_noisy)
                loss1 = tf.reduce_mean(tf.square(x_train - x_result))
                grads = tape.gradient(loss1, model1.trainable_variables)
            optimizer1.apply_gradients(zip(grads, model1.trainable_variables))
            if step % 2 == 0:
                print('\b\b\b\b\b\b\b', '%4.1f' % (100 * step / len_train_time), '%', end='')
        print(' ;LOSS值：', '%06f' % float(loss1), end='')
        if epoch1 % 5 == 0:
            model1.save_weights(path_model_weights + '_{0}'.format(epoch1))
            print(';第{0}次Model已储存'.format(epoch1), end='')
        print('')
    model1.save_weights(path_model_weights + '_final')
    # LenSegy.len_save_segy_multi_process(data_kernel2d_merge, path_segy_file, 'F:/test_LenKernel2D.segy')


if __name__ == '__main__':
    train()
