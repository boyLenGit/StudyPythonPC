# ------ Segy文件处理API ------
import numpy
import segyio
import multiprocessing


# 输入list和path即可。目前只能用与原文件同的规模的数据来生成segy，尚不可以用任意规模的数据生成segy
def len_save_segy(input_list, read_path, save_path):
    # ------------------------------ 数据初始化 ------------------------------
    input_list = numpy.array(input_list)
    input_len = len(input_list)
    # -------------------------------------------------------------------------
    with segyio.open(read_path) as src:
        print('√ 开始保存新Segy文件，原文件共{0}条数据，单条长{1}；新内容共{2}条数据，单条长{3}。'
              .format(len(src.trace), len(src.samples), input_len, len(input_list[0])), end='')
        # ------------------------------ Segy数据处理 ------------------------------
        header_list = []
        for i1 in range(input_len):
            header_list.append(src.header[i1])
        # -------------------------------------------------------------------------
        spec = segyio.spec()
        spec.sorting = src.sorting
        spec.format = src.format
        spec.samples = input_list[0]
        spec.ilines = src.ilines
        spec.xlines = src.xlines
        spec.tracecount = input_len
        print('SegyIO信息：src.tracecount={0}，src.samples长度={1}。'.format(src.tracecount, len(src.samples)))
        with segyio.create(save_path, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            dst.header = header_list
            dst.trace = input_list
    print('√ 提示：“{0}”文件已经写入完成'.format(save_path))


# 输入list和path即可。目前只能用与原文件同的规模的数据来生成segy，尚不可以用任意规模的数据生成segy
def len_save_segy_only_trace(input_list, read_path, save_path):
    """
    [功能]:针对地质信息缺失，需要ignore_geometry=True参数的情况而设立的函数。因为ignore_geometry=True会导致xline、iline无法读取而报错，因此该函数不会读取xline、iline
    :param input_list: 要保存的内容，通常为list或者numpy.array
    :param read_path: 读取文件的路径
    :param save_path: 保存文件的路径
    :return:
    """
    # ------------------------------ 数据初始化 ------------------------------
    input_list = numpy.array(input_list, dtype=numpy.float32)
    input_len = len(input_list)
    # -------------------------------------------------------------------------
    with segyio.open(read_path, ignore_geometry=True) as src:
        print('√ 开始保存新Segy文件，原文件共{0}条数据，单条长{1}；新内容共{2}条数据，单条长{3}。'
              .format(len(src.trace), len(src.samples), input_len, len(input_list[0])), end='')
        # ------------------------------ Segy数据处理 ------------------------------
        header_list = []
        for i1 in range(input_len):
            header_list.append(src.header[i1])
        # -------------------------------------------------------------------------
        spec = segyio.spec()
        spec.sorting = src.sorting
        spec.format = src.format
        spec.samples = input_list[0]
        spec.ilines = src.ilines
        spec.xlines = src.xlines
        spec.tracecount = input_len
        print('SegyIO信息：src.tracecount={0}，src.samples长度={1}。'.format(src.tracecount, len(src.samples)))
        with segyio.create(save_path, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            dst.header = header_list
            dst.trace = input_list
    print('√ 提示：“{0}”文件已经写入完成'.format(save_path))


def len_save_segy_multi_process(input_list, read_path, save_path):
    """
    将数据保存成为segy文件。
    :param input_list:输入要保存为segy的地震数据，应为二维list或numpy.array。数据格式必须为二维的(列, 行)，第一位存放各列，第二维存放每列的数据点
    :param read_path:参考segy模版的路径。
    :param save_path:待保存地震数据的路径。
    """
    progress1 = multiprocessing.Process(target=len_save_segy, name='001', args=(input_list, read_path, save_path))
    progress1.start()
    print('√ 多进程正在运行')


def len_save_segy_only_trace_multi_process(input_list, read_path, save_path):
    """
    [功能]:将数据保存成为segy文件（针对地质信息缺失版）。因为地质信息缺失，所以需要ignore_geometry=True参数的情况而设立的函数。因为ignore_geometry=True会导致xline、iline无法读取而报错，因此该函数不会读取xline、iline。
    :param input_list:输入要保存为segy的地震数据，应为二维list或numpy.array。数据格式必须为二维的(列, 行)，第一位存放各列，第二维存放每列的数据点
    :param read_path:参考segy模版的路径。
    :param save_path:待保存地震数据的路径。
    """
    progress1 = multiprocessing.Process(target=len_save_segy_only_trace, name='001', args=(input_list, read_path, save_path))
    progress1.start()
    print('√ 多进程正在运行')


def len_get_segy_max_value(input_list):
    max_value = 0
    for i1 in range(len(input_list)):
        for i2 in range(len(input_list[0])):
            max_value = input_list if input_list[i1][i2] > max_value else max_value
    print('√ 最大值为：', max_value)
    return max_value


def len_get_segy_average_max_value(input_list):
    max_value_fusion = 0
    len_input = len(input_list)
    print('√ 正在求平均最大值：     ', end='')
    for i1 in range(len(input_list)):
        max_value = 0
        for i2 in range(len(input_list[0])):
            max_value = input_list[i1][i2] if input_list[i1][i2] > max_value else max_value
        max_value_fusion += max_value
        if i1 % 100 == 0:
            print('\b\b\b\b\b', '%02d' % (100*i1//len_input), '%',  end='')
    average_max_value = max_value_fusion/len(input_list)
    print('√ 平均最大值为：', average_max_value)
    return average_max_value


def len_read_segy_file(read_path):
    """
    [功能]:地震数据的读取
    :param read_path: 读取文件的路径
    :return: 返回读取的地震数据内容，是一个dic格式
    """
    return_dic = {}
    with segyio.open(read_path, mode='r+') as SegyFile:
        # 将segy数据的trace提取出来,此时为ndarray格式数据（可以用numpy.array(data1)生成）
        return_dic['trace_raw'] = SegyFile.trace.raw[:]
        return_dic['text'] = SegyFile.text
        return_dic['tracecount'] = SegyFile.tracecount
        return_dic['trace'] = SegyFile.trace
        return_dic['header'] = SegyFile.header
        return_dic['samples'] = SegyFile.samples
        return_dic['xline'] = SegyFile.xline
        return_dic['xlines'] = SegyFile.xlines
        return_dic['iline'] = SegyFile.iline
        return_dic['ilines'] = SegyFile.ilines
    SegyFile.close()
    return return_dic


def len_read_segy_file_only_trace(read_path):
    """
    [功能]:针对地质信息缺失，需要ignore_geometry=True参数的情况而设立的函数。因为ignore_geometry=True会导致xline、iline无法读取而报错，因此该函数不会读取xline、iline
    :param read_path: 读取文件的路径
    :return: 返回读取的地震数据内容，是一个dic格式
    """
    return_dic = {}
    with segyio.open(read_path, mode='r+', ignore_geometry=True) as SegyFile:
        # 将segy数据的trace提取出来,此时为ndarray格式数据（可以用numpy.array(data1)生成）
        return_dic['trace_raw'] = SegyFile.trace.raw[:]
        return_dic['text'] = SegyFile.text
        return_dic['tracecount'] = SegyFile.tracecount
        return_dic['trace'] = SegyFile.trace
        return_dic['header'] = SegyFile.header
        return_dic['samples'] = SegyFile.samples
        # return_dic['xline'] = SegyFile.xline
        return_dic['xlines'] = SegyFile.xlines
        # return_dic['iline'] = SegyFile.iline
        return_dic['ilines'] = SegyFile.ilines
    SegyFile.close()
    return return_dic




