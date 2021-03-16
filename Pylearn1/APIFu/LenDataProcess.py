import numpy
import tensorflow as tf


def segy_to_tf_for_result_1output(data, to_be_one, reshape_dim, expand_axis, batch_size):
    """ 说明：将Segy格式的数据转换成为TensorFlow的Batch化的DataSet数据，维度调整为[batch, 数据量, 1] """
    data_output = numpy.array(data).copy()
    data_output = (data_output / to_be_one).astype(numpy.float32)
    data_output = tf.reshape(data_output, (-1, reshape_dim[1])) if reshape_dim is not None else data_output
    data_output = tf.expand_dims(data_output, axis=expand_axis)
    data_output = tf.data.Dataset.from_tensor_slices(data_output).batch(batch_size=batch_size)
    return data_output


def segy_to_tf_for_result_2output(data1, data2, to_be_one, reshape_dim, expand_axis, batch_size):
    """ 说明：将Segy格式的数据转换成为TensorFlow的Batch化的DataSet数据，维度调整为[batch, 数据量, 1] """
    data_output1, data_output2 = numpy.array(data1).copy(), numpy.array(data2).copy()
    data_output1, data_output2 = (data_output1 / to_be_one).astype(numpy.float32), (data_output2 / to_be_one).astype(numpy.float32)
    data_output1 = tf.reshape(data_output1, (-1, reshape_dim[1])) if reshape_dim is not None else data_output1
    data_output2 = tf.reshape(data_output2, (-1, reshape_dim[1])) if reshape_dim is not None else data_output2
    data_output1, data_output2 = tf.expand_dims(data_output1, axis=expand_axis), tf.expand_dims(data_output2, axis=expand_axis)
    dataset_output = tf.data.Dataset.from_tensor_slices((data_output1, data_output2)).batch(batch_size=batch_size)
    return dataset_output


def len_kernel_2d_cut(data_input, kernel_size=256, kernel_step=224):
    """ LenKernel2D版。用于将数据切割成二维块，用于训练用。
        返回值： kernel_1v_2v_fu_reshape返回LenKernel2D处理后的数据, i_1v降维前kernel的一维长度与二维长度，这两个用于还原原始规模数据用的
    """
    print('√ LenKernel切割器启动(Size={0},Step={1}) ;输入数据维度{2};'.format(kernel_size, kernel_step, data_input.shape), end='')
    kernel_1v_fu = helper_len_kernel_2d_cut(data_input, kernel_size, kernel_step)
    kernel_1v_cut_fu = []
    for kernel_1v in kernel_1v_fu:
        kernel_2v_fu = []
        for kernel_1v_single_col in kernel_1v:
            kernel_2v_fu.append(helper_len_kernel_2d_cut(kernel_1v_single_col, kernel_size, kernel_step))    # 2021.3.11修改点
        kernel_1v_cut_fu.append(kernel_2v_fu)
    print(' LenKernel2D处理后维度:{0};'.format(numpy.array(kernel_1v_cut_fu).shape), end='')
    kernel_1v_2v_fu_reshape = []
    for data1 in kernel_1v_cut_fu:  # 将单条(256, 2, 256)提取出来，翻转成(2, 256, 256)
        kernel_1v_2v_fu_reshape.append(len_exchange_1v_to_2v(data1))
    print(' 维度翻转后维度:{0};'.format(numpy.array(kernel_1v_2v_fu_reshape).shape), end='')
    dimension_kernels = [len(kernel_1v_2v_fu_reshape), len(kernel_1v_2v_fu_reshape[0])]
    kernel_1v_2v_fu_reshape = len_mix_1v_and_2v(kernel_1v_2v_fu_reshape)  # 合并维度(2764, 2, 256, 256)→(5528, 256, 256)
    print(' 合并维度后维度:{0};'.format(numpy.array(kernel_1v_2v_fu_reshape).shape))
    return kernel_1v_2v_fu_reshape, dimension_kernels


def helper_len_kernel_2d_cut(input1, size=256, step=224):
    len_data = len(input1)
    cnt = 0
    list_fusion = []
    while True:
        if cnt * step + size > len_data:
            list_fusion.append(input1[len_data - size: len_data])
            break
        list_fusion.append(input1[cnt * step: cnt * step + size])
        if cnt * step + size == len_data: break
        cnt += 1
    return list_fusion


def len_exchange_1v_to_2v(data1):
    """ 说明：将两个相邻的维度进行调换：(A维度, B维度)→(B维度, A维度) """
    new_1v = []
    for i2 in range(len(data1[0])):
        new_2v = []
        for i1 in range(len(data1)):
            new_2v.append(data1[i1][i2])
        new_1v.append(new_2v)
    return new_1v


def len_mix_1v_and_2v(data1):
    len_1v, len_2v = len(data1), len(data1[0])
    new_list = []
    for i1 in range(len_1v):
        for i2 in range(len_2v):
            new_list.append(data1[i1][i2])
    return new_list


def len_kernel_2d_merge(input_been_kernel, input_origin, dimension_kernels, size=256, step=224):
    cnt = 0
    len_1v_origin, len_2v_origin = len(input_origin), len(input_origin[0])
    # -------------------- 将降维过的矩阵升维 --------------------
    # 示例：(5528, 256, 256)→(2764, 2, 256, 256)
    up_dimension_fu = []
    for i1 in range(dimension_kernels[0]):
        up_dimension_2v_kernel = []
        for i2 in range(dimension_kernels[1]):
            up_dimension_2v_kernel.append(input_been_kernel[cnt])
            cnt += 1
        up_dimension_fu.append(up_dimension_2v_kernel)
    print('√ 升维后维度:{0};'.format(numpy.array(up_dimension_fu).shape), end='')
    # -------------------- (2764, 2, 256, 256)→(2764, 256, 2, 256) --------------------
    exchange_fu = [len_exchange_1v_to_2v(data1) for data1 in up_dimension_fu]
    print(' 换维后维度:{0};'.format(numpy.array(exchange_fu).shape), end='')
    # -------------------- (2764, 256, 2, 256)→(2764, 256, 462) --------------------
    merge_1v = helper_len_kernel_2d_merge(exchange_fu, len_1v_origin, size=size, step=step)
    print(' 1维融合后维度:{0};'.format(numpy.array(merge_1v).shape), end='')
    merge_2v = []
    for data2 in merge_1v:
        merge_2v.append(helper_len_kernel_2d_merge(data2, len_2v_origin, size=size, step=step))
    print(' 2维融合后维度:{0};'.format(numpy.array(merge_2v).shape), end='')
    return merge_2v


def len_kernel_2d_merge_by_add(input_been_kernel, input_origin, dimension_kernels, size=256, step=224):
    cnt = 0
    len_1v_origin, len_2v_origin = len(input_origin), len(input_origin[0])
    # -------------------- 将降维过的矩阵升维 --------------------
    # 示例：(5528, 256, 256)→(2764, 2, 256, 256)
    up_dimension_fu = []
    for i1 in range(dimension_kernels[0]):
        up_dimension_2v_kernel = []
        for i2 in range(dimension_kernels[1]):
            up_dimension_2v_kernel.append(input_been_kernel[cnt])
            cnt += 1
        up_dimension_fu.append(up_dimension_2v_kernel)
    print('√ 升维后维度:{0};'.format(numpy.array(up_dimension_fu).shape), end='')
    # -------------------- (2764, 2, 256, 256)->(2764, 256, 2, 256) --------------------
    exchange_fu = [len_exchange_1v_to_2v(data1) for data1 in up_dimension_fu]
    print(' 换维后维度:{0};'.format(numpy.array(exchange_fu).shape), end='')
    # -------------------- (2764, 256, 2, 256)->(2764, 256, 462) --------------------
    merge_1v = helper_len_kernel_2d_merge_by_add_plus(exchange_fu, len_1v_origin, size=size, step=step)
    print(' 1维融合后维度:{0};'.format(numpy.array(merge_1v).shape), end='')
    merge_2v = []
    for data2 in merge_1v:
        merge_2v.append(helper_len_kernel_2d_merge_by_add_plus(data2, len_2v_origin, size=size, step=step))
    print(' 2维融合后维度:{0};'.format(numpy.array(merge_2v).shape), end='')
    return merge_2v


def helper_len_kernel_2d_merge(input_data, len_origin, size=256, step=224):
    merge_fu = []
    distance = int((size - step) / 2)
    len_input = len(input_data)
    for i1 in range(len_input):
        if i1 == len_input - 1:
            temp_merge = input_data[i1][len(input_data[i1]) - (len_origin - step * i1 - distance): len_origin]
            merge_fu.extend(temp_merge)
            break
        if i1 == 0:
            temp_merge = input_data[i1][0: step + distance]
        else:
            temp_merge = input_data[i1][distance: step + distance]
        merge_fu.extend(temp_merge)
    return merge_fu


def helper_len_kernel_2d_merge_by_add(input_data, len_origin, size=256, step=224):
    """
    [过程] (a,b,多余维度)->(c,,多余维度)
    :param input_data: 输入的是二维list（实际维度随便，但有效维度就只有2维）：(a,b,多余维度)->(c,,多余维度)
    :param len_origin: merge_fu的长度，也就是原数据的长度
    :param size:
    :param step:
    :return:
    """
    merge_fu = []
    distance, distance_half = int(size - step), int((size - step) / 2)
    len_input_1v, len_input_2v = len(input_data), len(input_data[0])
    for i1 in range(len_input_1v):
        if i1 == len_input_1v - 2:  # 如果i1位于最后一个位置
            area_this, area_last = input_data[i1][0:distance], input_data[i1 - 1][len_input_2v - distance:len_input_2v]
            temp_area_merge = ((numpy.array(area_this) + numpy.array(area_last)) / 2).tolist()
            if (len_origin - size - i1*step) <= distance:
                temp_merge = temp_area_merge[0:len_origin - size - i1*step]
                merge_fu.extend(temp_merge)
                continue
            temp_merge = temp_area_merge
            temp_merge.extend(input_data[i1][distance:len_origin - size - i1*step])
            merge_fu.extend(temp_merge)
            # print('& 位置{0} 维度:{1}'.format(i1, numpy.array(merge_fu).shape), '倒2', len_origin - size - i1*step)
            continue
        if i1 == len_input_1v - 1:  # 如果i1位于最后一个位置
            distance_remain = len(input_data[i1]) - (len_origin - step * i1 - distance_half) + distance_half
            area_this, area_last = input_data[i1][0:distance_remain], input_data[i1 - 1][len_input_2v - distance_remain:len_input_2v]
            # print('Shape:', numpy.array(area_this).shape, numpy.array(area_last).shape)
            temp_area_merge = ((numpy.array(area_this) + numpy.array(area_last)) / 2).tolist()
            temp_merge = temp_area_merge
            temp_merge.extend(input_data[i1][distance_remain:len_origin])
            merge_fu.extend(temp_merge)
            # print('& 位置{0} 维度:{1}'.format(i1, numpy.array(merge_fu).shape), '倒1', distance_remain)
            break
        if i1 == 0:  # 刚开始时，起始索引需要为0
            temp_merge = input_data[i1][0:step]
        else:
            area_this, area_last = input_data[i1][0:distance], input_data[i1 - 1][len_input_2v - distance:len_input_2v]
            temp_area_merge = ((numpy.array(area_this) + numpy.array(area_last)) / 2).tolist()
            temp_merge = temp_area_merge
            temp_merge.extend(input_data[i1][distance:step])
        merge_fu.extend(temp_merge)
        # print('& 位置{0} 维度:{1}'.format(i1, numpy.array(merge_fu).shape))
    return merge_fu


def helper_len_kernel_2d_merge_by_add_plus(input_data, len_origin, size=256, step=224):
    """

    :param input_data:
    :param len_origin:
    :param size:
    :param step:
    :return:
    """
    merge_fu = []
    overlap, overlap_half = int(size - step), int((size - step) / 2)
    len_input_1v, len_input_2v = len(input_data), len(input_data[0])
    for i1 in range(len_input_1v):
        input_data[i1] = input_data[i1].tolist() if type(input_data[i1]) is numpy.ndarray else input_data[i1]
        if i1 == 0:
            merge_fu = input_data[i1]
            continue
        if i1 == len_input_1v - 1:
            overlap_end = size - (len_origin - len(merge_fu))
            area_last, area_this = merge_fu[-overlap_end:], input_data[i1][:overlap_end]
            del merge_fu[-overlap_end:]
            area_merged = helper__len_kernel_2d_merge_by_add_plus__merger_by_multiply(this=area_this, last=area_last)
            merge_fu.extend(area_merged)
            merge_fu.extend(input_data[i1][overlap_end:])
            break
        area_last, area_this = merge_fu[-overlap:], input_data[i1][:overlap]
        del merge_fu[-overlap:]
        area_merged = helper__len_kernel_2d_merge_by_add_plus__merger_by_multiply(this=area_this, last=area_last)
        merge_fu.extend(area_merged)
        merge_fu.extend(input_data[i1][overlap:])
    return merge_fu


def helper__len_kernel_2d_merge_by_add_plus__merger_by_multiply(this, last):
    """
    [功能] 辅助函数，将地震数据块的重叠部分进行加权融合，从而避免块边缘的合成痕迹明显的问题
    :param this:当前块的重叠区域
    :param last:上一个块的重叠区域
    :return:加权融合后的块
    """
    len_1v = len(this)
    # 构建权重矩阵
    for i1 in range(len_1v):
        this[i1] = (numpy.array(this[i1]) * ((i1 + 1) / (len_1v + 1)))
        last[i1] = (numpy.array(last[i1]) * ((len_1v - i1) / (len_1v + 1)))
        # print((i1 + 1), (len_1v - i1), ((i1 + 1) / (len_1v + 1)), ((len_1v - i1) / (len_1v + 1)))
    return (numpy.array(this) + numpy.array(last)).tolist()




