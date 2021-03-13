# 实现论文内容
# ***说明***
# 改模型测试了不同网络结构下的效果，证明去掉池化、上采样的loss最小；效果排名：去掉池化上采样、1234BN > 1234BN > 0BN > 不带BN的原始网络

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import random
import APIFu.LenSegy as LenSegy
import APIFu.LenModels.Conv1D_AE as Conv1D_AE
import APIFu.LenDataProcess as LenDataProcess

len_kernel_size = 256
input_data_second_dim, input_data_third_dim = len_kernel_size, 1  # 输入网络的维度
conv_kernel_size_mnist, conv_kernel_size_ori = [28, 18, 10, 18, 28], [130, 65, 20, 65, 130]
conv_kernel_size = conv_kernel_size_ori
to_be_one = 10000.  # 地震数据归一化参数，用于将地震数据约束在-1~1之间
segy_max_value = 17024  # Max: 619101 ;average:10024.817490199499
plot_show_num = 200  # 设置Plot绘图的数据条数，与batch紧密相关。
len_origin_single_data = 0  # 原始数据单条的长度
sample_step = 64 * 3  # 将单列数据分割成多组len_kernel_size规模数据的采样步长
path_model = '/TensorFlowFu/TensorTeacher/data/001_Conv1DAE_v6.1.Len'
path_model_weights = 'D:/boyLen/Python/Pylearn1/TensorFlowFu/TensorTeacher/data/001_Conv1DAE_v6.3_40'
path_segy_file = 'E:/Research/data/F3_entire.segy'
path_segy_mix_noisy_file = 'E:/Research/data/ArtificialData/test_Mix_Noisy_v2.segy'


# path_segy_file = 'E:/Research/data/CMP_INL2000_stk_org_gain.SEGY'


def get_data_v4_segy():
    segy_data = LenSegy.len_read_segy_file(path_segy_file)['trace_raw']
    segy_data_mix_noisy = LenSegy.len_read_segy_file(path_segy_mix_noisy_file)['trace_raw']
    # 初始化数据
    global len_origin_single_data
    len_origin_single_data = len(segy_data[0])
    # ↓ segy的列数、行数调整，以适应网络模型（单条数据的长度必须是4的倍数，因为有两次2维池化和2维上采样）
    # LenSegy.len_get_segy_average_max_value(segy_data)  # 求每列的平均最大值
    # ------ test ------
    # ------
    # LenKernel处理核心
    x_, x_mix_noisy = numpy.array(
        boylen_kernel_generator_for_train_2input(segy_data.copy(), segy_data_mix_noisy.copy()))
    print('@ 经LenKernel后的数据规模：', x_.shape, x_mix_noisy.shape)
    # Train、Test数据范围选取
    x_train, x_train_mix_noisy = x_[:2000000], x_mix_noisy[:2000000]  # 这两个索引数必须一致
    x_test, x_test_mix_noisy = segy_data[-10000:], segy_data_mix_noisy[-10000:]  # 这两个索引数必须一致
    # 数据类型转化与缩放
    x_train, x_train_mix_noisy = (x_train / to_be_one).astype(numpy.float32), (x_train_mix_noisy / to_be_one).astype(
        numpy.float32)
    x_test, x_test_mix_noisy = (x_test / to_be_one).astype(numpy.float32), (x_test_mix_noisy / to_be_one).astype(
        numpy.float32)
    # 维度变形
    x_train, x_train_mix_noisy = tf.reshape(x_train, (-1, input_data_second_dim)), tf.reshape(x_train_mix_noisy, (
    -1, input_data_second_dim))
    x_test, x_test_mix_noisy = tf.reshape(x_test, (-1, len_origin_single_data)), tf.reshape(x_test_mix_noisy, (
    -1, len_origin_single_data))
    x_train, x_train_mix_noisy = tf.expand_dims(x_train, axis=2), tf.expand_dims(x_train_mix_noisy, axis=2)
    x_test, x_test_mix_noisy = tf.expand_dims(x_test, axis=2), tf.expand_dims(x_test_mix_noisy, axis=2)
    # Batch化。x_test_all_data为全部的原始数据，x_test_all_noisy为全部加了噪声的原始数据
    x_train_dataset, x_test_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train_mix_noisy)).batch(
        batch_size=plot_show_num) \
        , tf.data.Dataset.from_tensor_slices((x_test, x_test_mix_noisy)).batch(batch_size=plot_show_num)
    return x_train_dataset, x_test_dataset


def boylen_kernel_generator_for_train_2input(input_list1, input_list2):  # 【核采样】指定宽度采样法（仅对x_train进行处理即可）
    enlarge_data_list1, enlarge_data_list2 = [], []
    len_input_list = len(input_list1)
    len_input_list_single = len(input_list1[0])
    print('@ LenKernel处理(train集)：       ', end='')
    for i1 in range(len_input_list):
        one_trace_cut_times = int(10 * random.random())
        step_add_all = 0
        for i2 in range(one_trace_cut_times):
            step_single = int(random.random() * (len_input_list_single - len_kernel_size - 1) / 3)
            step_add_all += step_single
            if (step_add_all + len_kernel_size - 1) >= len(input_list1[0]): break
            enlarge_data_list1.append(input_list1[i1][step_add_all: step_add_all + len_kernel_size])
            enlarge_data_list2.append(input_list2[i1][step_add_all: step_add_all + len_kernel_size])
        if i1 % 20 == 0:
            print('\b\b\b\b\b\b\b', '%4.1f' % (100 * i1 / len_input_list), '%', end='')
    print('')
    return enlarge_data_list1, enlarge_data_list2


def boylen_train(db_train, db_test, train_time=100):
    # 初始化数据
    global len_origin_single_data
    len_train_time = len(db_train)
    # ------------------------------------- 初始化并训练模型 ---------------------------------------
    model1 = Conv1D_AE.no_poolsampling(conv_kernel_size=conv_kernel_size, input_data_third_dim=input_data_third_dim)
    model1.build(input_shape=(None, None, input_data_third_dim))
    model1.load_weights('D:/boyLen/Python/Pylearn1/TensorFlowFu/TensorTeacher/data/001_Conv1DAE_v6.3_45s')
    optimizer1 = tf.optimizers.Adam(lr=0.001, name='SGD')
    for epoch1 in range(45, train_time):
        print('@ 第{0}次训练：       '.format(epoch1 + 1), end='')
        for step, (x_train, x_train_noisy) in enumerate(db_train):
            with tf.GradientTape() as tape:
                x_result = model1(x_train_noisy)
                loss1 = tf.reduce_mean(tf.square(x_train - x_result))
                grads = tape.gradient(loss1, model1.trainable_variables)
            optimizer1.apply_gradients(zip(grads, model1.trainable_variables))
            if step % 5 == 0:
                print('\b\b\b\b\b\b\b', '%4.1f' % (100 * step / len_train_time), '%', end='')
        print(' ;LOSS值：', '%06f' % float(loss1))
        if epoch1 % 5 == 0: model1.save_weights(path_model_weights + '_{0}'.format(epoch1))
        # if epoch1 % 2 != 0 or epoch1 != 0: continue
        # ↓ 测试模型结果测试
        x_test_list, x_test_result_list, x_test_noisy_list = [], [], []
        for step, (x_test, x_test_noisy) in enumerate(db_test):
            x_test_result = model1(x_test_noisy)
            x_test_list.append(x_test)
            x_test_result_list.append(x_test_result)
            x_test_noisy_list.append(x_test_noisy)
        x_test, x_test_result, x_test_noisy = tf.reshape(x_test_list, [-1, len_origin_single_data]), tf.reshape(
            x_test_result_list, [-1, len_origin_single_data]), tf.reshape(x_test_noisy_list,
                                                                          [-1, len_origin_single_data])
        x_test, x_test_result, x_test_noisy = x_test.numpy(), x_test_result.numpy(), x_test_noisy.numpy()
        x_test, x_test_result, x_test_noisy = x_test * to_be_one, x_test_result * to_be_one, x_test_noisy * to_be_one
        # 绘图部分
        if epoch1 == 0:
            boylen_plot(x_test, title='Input')
            boylen_plot(x_test_noisy, title='Input_noisy')
        if epoch1 % 5 == 0:
            boylen_plot(x_test_result, title='Result{0}'.format(epoch1))
    LenSegy.len_save_segy_multi_process(x_test_noisy, path_segy_file, 'J:/test_noisy.segy')
    LenSegy.len_save_segy_multi_process(x_test_result, path_segy_file, 'F:/test_result.segy')
    model1.save_weights(path_model_weights + '_final')


def len_forecast():
    path_segy_file_len_forecast = 'E:/Research/data/CMP_INL2000_stk_org_gain.SEGY'
    segy_data = LenSegy.len_read_segy_file(path_segy_file_len_forecast)['trace_raw']
    segy_data_dataset = LenDataProcess.segy_to_tf_for_result_1output(segy_data, None, to_be_one, [0, input_data_second_dim, input_data_third_dim], 2, plot_show_num)
    model1 = Conv1D_AE.no_poolsampling(conv_kernel_size=conv_kernel_size, input_data_third_dim=input_data_third_dim)
    model1.build(input_shape=(None, None, input_data_third_dim))
    model1.load_weights('D:/boyLen/Python/Pylearn1/TensorFlowFu/TensorTeacher/data/001_Conv1DAE_v6.3_45s')

    # ------------------------------------- 生成训练完预测出来的segy数据 ---------------------------------------
    list_forecast_all_fusion, list_input_all_fusion = [], []
    len_data_all = len(segy_data_dataset)
    print('@ 训练完毕，开始生成全规模数据：     ', end='')
    for step, x_all in enumerate(segy_data_dataset):
        x_all_forecast_single = model1(x_all)
        list_forecast_all_fusion.extend(x_all_forecast_single)
        list_input_all_fusion.extend(x_all)
        if step % 5 == 0:
            print('\b\b\b\b\b\b\b', '%4.1f' % (100 * step / len_data_all), '%', end='')
    print('')

    # ------------------------------------- 预测数据单条全部融合完毕，画个图看看效果 ---------------------------------------
    print('@ shape:', tf.shape(list_forecast_all_fusion), tf.shape(list_input_all_fusion))
    # 维度改变
    list_forecast_all_fusion, list_input_all_fusion = tf.reshape(list_forecast_all_fusion, [-1, len(list_forecast_all_fusion[1])]), \
                                                      tf.reshape(list_input_all_fusion, [-1, len(list_input_all_fusion[1])])
    # 数据转化，从(-1,1)扩张到原始大小
    list_forecast_all_fusion, list_input_all_fusion = (list_forecast_all_fusion.numpy()) * to_be_one, (list_input_all_fusion.numpy()) * to_be_one
    # 绘图，看效果用的
    boylen_plot(numpy.array(list_input_all_fusion), title='Original All Fusion')
    boylen_plot(numpy.array(list_forecast_all_fusion), title='Result All Fusion')
    # 多进程同步保存Segy文件
    LenSegy.len_save_segy_multi_process(list_forecast_all_fusion, path_segy_file_len_forecast, 'F:/test_result.segy')


def boylen_plot(input_list, line_num=plot_show_num, title='title'):
    list_x = []
    plt.figure(figsize=(12, 12))
    for i0 in range(len(input_list[0])):  # ← 生成x轴的标签数据
        list_x.append(i0)
    for i1 in range(line_num):
        for i2 in range(len(input_list[0])):
            input_list[i1][i2] += 30000 * i1
    for i3 in range(line_num):
        upper, lower = 3000 + i3 * 30000, -3000 + i3 * 30000
        big_list = numpy.ma.masked_where(input_list[i3] <= upper - 500, input_list[i3])
        mid_list = numpy.ma.masked_where((input_list[i3] <= lower - 3000) | (input_list[i3] >= upper + 3000),
                                         input_list[i3])
        lit_list = numpy.ma.masked_where(input_list[i3] >= lower + 500, input_list[i3])
        plt.plot(big_list, list_x, 'b-', mid_list, list_x, 'r-', lit_list, list_x, 'b-', linewidth=2)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    '''datasets_train, datasets_test = get_data_v4_segy()
    boylen_train(db_train=datasets_train, db_test=datasets_test)'''
    len_forecast()
