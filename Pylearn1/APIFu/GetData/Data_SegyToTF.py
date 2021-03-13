import numpy
import tensorflow as tf

import APIFu.LenSegy as LenSegy


def rt_forecast_dataset_1v(path_segy_file, to_be_one, batch_size):
    segy_data = LenSegy.len_read_segy_file(path_segy_file)['trace_raw']
    # 初始化数据
    len_origin_single_data = len(segy_data[0])
    # ↓ segy的列数、行数调整，以适应网络模型（单条数据的长度必须是4的倍数，因为有两次2维池化和2维上采样）
    x_test_all_data = segy_data[:]  # 复制原数据
    # 数据类型转化与缩放
    x_test_all_data = (x_test_all_data / to_be_one).astype(numpy.float32)
    # 维度变形
    x_test_all_data = tf.expand_dims(x_test_all_data, axis=2)
    # Batch化。x_test_all_data为全部的原始数据，x_test_all_noisy为全部加了噪声的原始数据
    x_test_all_dataset = tf.data.Dataset.from_tensor_slices(x_test_all_data).batch(batch_size=batch_size)
    return x_test_all_dataset
