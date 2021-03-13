import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential
import numpy
import segyio
import matplotlib.pyplot as plt
# import random
import APIFu.LenSegy as LenSegy
import APIFu.LenModels.Conv2D_AE as Conv2D_AE
import APIFu.LenDataProcess as LenDataProcess


"""初始化变量"""
dimension2d_input = 256
read_num = 200000
to_be_one = 10000.
path_model_weights_save = 'D:/boyLen/Python/Pylearn1/TensorFlowFu/TensorTeacher/data/05_Conv2DAE/v16'
path_model_weights_load = 'D:/boyLen/Python/Pylearn1/TensorFlowFu/TensorTeacher/data/05_Conv2DAE/v16_final'

path_segy_file_label = 'E:/Research/data/ArtificialData/extract_segy_valid_line_v2.segy'
path_segy_file_be_forecast = 'E:/Research/data/CMP_INL2000_stk_org_gain.SEGY'
path_segy_file_train = 'E:/Research/data/ArtificialData/test_Mix_Noisy_v3.segy'


def train(train_time=100):
    # ------------------------------------- 获取并处理数据 ---------------------------------------
    data_origin = LenSegy.len_read_segy_file(path_segy_file_label)['trace_raw'][:read_num]
    data_origin_noisy = LenSegy.len_read_segy_file(path_segy_file_train)['trace_raw'][:read_num]
    data_kernel2d_cut, dimension_kernels = LenDataProcess.len_kernel_2d_cut(data_origin)
    data_kernel2d_cut_noisy, dimension_kernels_noisy = LenDataProcess.len_kernel_2d_cut(data_origin_noisy)
    dataset = LenDataProcess.segy_to_tf_for_result_2output(data1=data_kernel2d_cut, data2=data_kernel2d_cut_noisy,
                                                           to_be_one=to_be_one, reshape_dim=None, expand_axis=3,
                                                           batch_size=8)
    # ------------------------------------- 训练模型 ---------------------------------------
    model1 = Conv2D_AE.Conv2DAE_GoogleNet_v2()
    model1.build(input_shape=(None, dimension2d_input, dimension2d_input, 1))  # [None, 256, 256, 1]
    model1.load_weights(path_model_weights_load)
    # model1.summary()
    len_train_time = len(dataset)
    optimizer1 = tf.optimizers.Adam(lr=0.001, name='SGD')
    for epoch1 in range(46, train_time):
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
            model1.save_weights(path_model_weights_save + '_{0}'.format(epoch1))
            print('Tips:已保存模型参数', end='')
        print('')
    model1.save_weights(path_model_weights_save + '_final')


def forecast():
    data_origin = LenSegy.len_read_segy_file(path_segy_file_be_forecast)['trace_raw'][:]
    data_kernel2d_cut, dimension_kernels = LenDataProcess.len_kernel_2d_cut(data_origin)
    dataset = LenDataProcess.segy_to_tf_for_result_1output(data=data_kernel2d_cut, to_be_one=to_be_one, reshape_dim=None, expand_axis=3, batch_size=8)
    model1 = Conv2D_AE.Conv2DAE_GoogleNet_v2()
    model1.build(input_shape=(None, dimension2d_input, dimension2d_input, 1))  # [None, 256, 256, 1]
    model1.load_weights(path_model_weights_load)
    model1.summary()
    data_fusion = []
    for step, x_train in enumerate(dataset):
        data_fusion.extend(model1(x_train))
    data_fusion = tf.reshape(data_fusion, [-1, dimension2d_input, dimension2d_input])
    data_fusion = (data_fusion.numpy()) * to_be_one
    data_fusion = LenDataProcess.len_kernel_2d_merge(data_fusion, data_origin, dimension_kernels)
    LenSegy.len_save_segy_multi_process(data_fusion, path_segy_file_be_forecast, 'F:/test_result_v4.segy')


if __name__ == '__main__':
    # train()
    forecast()
