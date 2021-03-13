from abc import ABC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential
import numpy
import segyio
import matplotlib.pyplot as plt
import random
import APIFu.LenSegy as LenSegy
import APIFu.LenModels.Conv2D_AE as Conv2D_AE
import APIFu.LenDataProcess as LenDataProcess

"""初始化变量"""
dimension2d_input = 256
read_num = 100000
to_be_one = 10000.
path_model_weights = 'D:/boyLen/Python/Pylearn1/TensorFlowFu/TensorTeacher/data/005_Conv2DAE_v1.2'
path_model_weights_load = 'D:/boyLen/Python/Pylearn1/TensorFlowFu/TensorTeacher/data/005_Conv2DAE_v1.2_final'

path_segy_file = 'E:/Research/data/F3_entire.segy'
path_segy_file_cmp = 'E:/Research/data/CMP_INL2000_stk_org_gain.SEGY'
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
    # ------------------------------------- 获取并处理数据 ---------------------------------------
    data_origin, data_origin_noisy = LenSegy.len_read_segy_file(path_segy_file)['trace_raw'][:read_num], \
                                     LenSegy.len_read_segy_file(path_segy_mix_noisy_file)['trace_raw'][:read_num]
    data_kernel2d_cut, dimension_kernels = LenDataProcess.len_kernel_2d_cut(data_origin)
    data_kernel2d_cut_noisy, dimension_kernels_noisy = LenDataProcess.len_kernel_2d_cut(data_origin_noisy)
    dataset = LenDataProcess.segy_to_tf_for_result_2output(data1=data_kernel2d_cut, data2=data_kernel2d_cut_noisy,
                                                           to_be_one=to_be_one, reshape_dim=None, expand_axis=3,
                                                           batch_size=8)
    # ------------------------------------- 训练模型 ---------------------------------------
    # model1 = Conv2DAe()
    model1 = Conv2D_AE.simple_cnn_ae()
    model1.build(input_shape=(None, dimension2d_input, dimension2d_input, 1))  # [None, 256, 256, 1]
    len_train_time = len(dataset)
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
        print(' ;LOSS值：', '%06f' % float(loss1))
        if epoch1 % 5 == 0: model1.save_weights(path_model_weights + '_{0}'.format(epoch1))
    model1.save_weights(path_model_weights + '_final')


def forecast():
    data_origin = LenSegy.len_read_segy_file(path_segy_file_cmp)['trace_raw'][:]
    data_kernel2d_cut, dimension_kernels = LenDataProcess.len_kernel_2d_cut(data_origin)
    dataset = LenDataProcess.segy_to_tf_for_result_1output(data=data_kernel2d_cut, to_be_one=to_be_one, reshape_dim=None, expand_axis=3, batch_size=8)
    model1 = Conv2DAe()
    model1.build(input_shape=(None, dimension2d_input, dimension2d_input, 1))  # [None, 256, 256, 1]
    model1.load_weights(path_model_weights_load)
    data_fusion = []
    for step, x_train in enumerate(dataset):
        data_fusion.extend(model1(x_train))
    data_fusion = tf.reshape(data_fusion, [-1, dimension2d_input, dimension2d_input])
    data_fusion = (data_fusion.numpy()) * to_be_one
    data_fusion = LenDataProcess.len_kernel_2d_merge(data_fusion, data_origin, dimension_kernels)
    # data_kernel2d_merge = LenDataProcess.len_kernel_2d_merge(data_kernel2d_cut, data_origin, dimension_kernels)
    LenSegy.len_save_segy_multi_process(data_fusion, path_segy_file_cmp, 'F:/test_result_v4.segy')


if __name__ == '__main__':
    # train()
    forecast()
