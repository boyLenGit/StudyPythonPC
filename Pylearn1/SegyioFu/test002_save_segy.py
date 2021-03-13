import segyio



def learn4():
    path2 = 'E:/Research/data/F3_entire.segy'
    path2_1 = 'E:/Research/data/F3_entire_test.segy'
    path3 = 'E:/Research/data/viking_small.segy'
    with segyio.open(path2, mode='r', ignore_geometry=True) as segy_file:
        with segyio.open(path2_1, mode='r+', ignore_geometry=True) as segy_file2:
            # print(len(segy_file2.trace))
            print(segy_file.trace[0])
            segy_file2.trace = segy_file.trace[0:100]
            segy_file2.flush()
        segy_file.close()
        # assert segyio.dt(segy_file2)


def learn5():
    path2 = 'E:/Research/data/F3_entire.segy'
    path3 = 'E:/Research/data/test1.segy'
    with segyio.open(path2) as src:
        spec = segyio.spec()
        spec.sorting = src.sorting
        spec.format = src.format
        spec.samples = src.samples
        spec.ilines = src.ilines
        spec.xlines = src.xlines
        spec.tracecount = src.tracecount
        print(len(src.samples))
        with segyio.create(path3, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            dst.header = src.header
            dst.trace = src.trace


def learn6():
    srcpath = 'E:/Research/data/F3_entire.segy'
    dstpath = 'F:/test1.segy'
    with segyio.open(srcpath) as src:
        spec = segyio.spec()
        spec.sorting = src.sorting
        spec.format = src.format
        spec.samples = src.samples[:len(src.samples) - 50]
        spec.ilines = src.ilines
        spec.xline = src.xlines
        with segyio.create(dstpath, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            dst.header = src.header
            dst.trace = src.trace


def learn7():
    read_path = 'E:/Research/data/F3_entire.segy'
    save_path = 'F:/test1.segy'
    col_num = 1000
    with segyio.open(read_path) as src:
        # ----- 输入数据处理
        input_list = src.trace.raw[:col_num]
        header_list = []
        for i1 in range(col_num):
            header_list.append(src.header[i1])
        # -----
        print('@ 开始保存新Segy文件，原文件共{0}条数据，单条长{1}；新内容共{2}条数据，单条长{3}。'
              .format(len(src.trace), len(src.samples), len(input_list), len(input_list[0])), end='')
        spec = segyio.spec()
        spec.sorting = src.sorting
        spec.format = src.format
        spec.samples = input_list[0]
        spec.ilines = src.ilines
        spec.xlines = src.xlines
        spec.tracecount = len(input_list)
        print('SegyIO信息：src.tracecount={0}，src.samples长度={1}。'.format(src.tracecount, len(src.samples)))
        with segyio.create(save_path, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            dst.header = header_list
            dst.trace = input_list
    print('@ 提示：“{0}”文件已经写入完成'.format(save_path))


if __name__ == '__main__':
    learn7()
