#  老师指导下的Spark编程验证
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import types
from pyspark.sql import functions
from pyspark.sql import SQLContext
import pyspark
import databricks
import pyspark.sql as sql
import matplotlib.pyplot as plt
from bokeh import charts
from bokeh.io import output_notebook
import segyio
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from segyio import TraceField

sc = SparkContext("local", "first app")
ss = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value") \
    .master('local[4]').getOrCreate()

global_df = None

# CSDN的一个读取segy文件的示例代码
def learn1():
    file_path = 'F:/Research/data/F3_entire.segy'
    with segyio.open(file_path) as segy_readfile1:
        # 设置内存映射文件快速读取(特别是如果文件很大…)
        segy_readfile1.mmap()
        # 打印二进制文件的头信息
        print('打印二进制文件的头信息:')
        print(segy_readfile1.bin, segy_readfile1.bin[segyio.BinField.Traces])
        # 读取前10个字节的道头
        print('读取前10个字节的道头:')
        print(segy_readfile1.header[10][segyio.TraceField.INLINE_3D])
        # 打印行内轴和横轴
        print('打印行内轴和横轴:')
        print(segy_readfile1.xlines)
        print(segy_readfile1.ilines)


# Segyio官方的第一个Tutorial代码（将原版的plt部分进行了修改）
def learn2():
    path1 = 'F:/Research/data/viking_small.segy'
    file_segy1 = segyio.open(path1, ignore_geometry=True)
    v_min, v_max = -(1e+2), 1e+2
    # trace：设定在trace模式下与segy数据交互
    data_imshow = file_segy1.trace.raw[:].T
    plt.imshow(X=data_imshow, cmap=plt.cm.seismic, vmin=v_min, vmax=v_max)
    print(len(data_imshow), data_imshow)
    plt.colorbar()
    plt.show()
    print(file_segy1.ilines)
    file_segy1.close()


def learn3():
    path1 = 'F:/Research/data/viking_small.segy'
    path2 = 'F:/Research/data/F3_entire.segy'
    print('Path1文件测试：')
    with segyio.open(filename=path1, mode='r+', ignore_geometry=True) as file1:
        print(file1.ilines)
    print(file1)
    file1.close()
    print('Path2文件测试：')
    with segyio.open(filename=path2, mode='r+') as file2:
        print(file2.ilines)
    print('file:')
    print(file2)
    file2.close()


def learn4():
    with segyio.open('F:/Research/data/viking_small.segy', mode='r+', ignore_geometry=True) as viking_small:
        spec = segyio.spec()
        spec.ilines = [1, 2, 3, 4]
        spec.xlines = [11, 12, 13]
        # spec.samples = list(range(50))
        spec.samples = viking_small.samples[:len(viking_small.samples) - 50]  # shorten all traces by 50 samples
        spec.sorting = viking_small.sorting
        spec.format = viking_small.format
        path1 = 'F:/Research/data/test.segy'
        with segyio.create(path1, spec) as file1:
            file1.text[0] = viking_small.text[0]
            file1.bin = viking_small.bin
            file1.header = viking_small.header
            file1.trace = viking_small.trace
            file1.close()


# SegyFile验证
def learn5():
    path2 = 'E:/Research/data/F3_entire.segy'
    with segyio.open(path2, mode='r+') as F3_entire:
        print('######attributes:', F3_entire.attributes('traces'))
        print('######flush:', F3_entire.flush())
        print('######mmap:', F3_entire.mmap())
        print('######dtype:', F3_entire.dtype)
        print('######fast:', F3_entire.fast)
        print('######iline:', F3_entire.iline)
        print('######ilines--len:', len(F3_entire.ilines))
        print('######xline:', F3_entire.xline)
        print('######xlines--len:', len(F3_entire.xlines))
        print('######trace:', F3_entire.trace)
        print('######trace.raw:', F3_entire.trace.raw[:])
        print('######header:', F3_entire.header)
        print('######header_len:', len(F3_entire.header))
        print('######header[0]:', F3_entire.header[0][TraceField.offset])
        print('######header[0].:', F3_entire.header[0][TraceField.offset])
        # print('######header.iline:', F3_entire.header.iline.header.segy)  # 返回了SegyFile、inlines、crosslines、traces、samples等。全是數字
        # print('######header.xline.lines:', F3_entire.header.xline.lines)  # 返回了一大串數字
        # print('######header.xline:', F3_entire.header.xline.header.segy)
        print('######text:', F3_entire.text)
        '''for i1 in range(len(F3_entire.text)):
            print('######For{0}:'.format(i1), F3_entire.text[i1])'''


#将地震数据转换成为RDD和DataFrame
def learn6():
    path2 = 'F:/Research/data/F3_entire.segy'
    path3 = 'F:/Research/data/viking_small.segy'
    with segyio.open(path2, mode='r+') as F3_entire:
        # 将segy数据的trace提取出来
        data1 = F3_entire.trace.raw[:]  # raw[:2000]可以限制读取条数为2000条
        print('SEGY数据条数：', len(data1))
        # 将segy的trace数据转换成为list数据
        list1 = []
        for i in range(3000):
            list_mini1 = []
            for ii in range(len(data1[0])):
                # 将数据类型强制转换成为float格式，因为DataFrame不支持numpy.float格式的数据(不支持numpy数据)。如果不转则报错。
                # 必须是将list每一项逐项转换，不支持将list整体都转换为float格式数据
                list_mini1.append(float(data1[i][ii]))
            list1.append(list(list_mini1))
        # 生成DataFrame的标签schema。应为string格式。
        schema1 = []
        for i in range(len(list1[0])):
            schema1.append("{}".format(i))
        # 创建RDD
        print('LIST:', list1)
        rdd1 = sc.parallelize(list1)
        print('RDD:', rdd1.collect())
        # 创建DataFrame
        df1 = ss.createDataFrame(data=list1, schema=schema1)
        df1.show(5)
        # RDD-->DF
        df2 = ss.createDataFrame(data=rdd1, schema=schema1)
        df2.show(5)
    F3_entire.close()


def learn7():
    # 每一列的数据格式必须相同，否则会报错
    list1 = [[1., 144.5, 5.9, 33., 77.], [2., 167.2, 5.4, 45., 99.], [3., 124.1, 5.2, 23., 55.]]
    rdd1 = sc.parallelize(c=list1)
    print('LIST:', list1)
    print('RDD:', rdd1.collect())
    schema1 = ['id', 'weight', 'height', 'age', 'gender']
    df1 = ss.createDataFrame(list1, schema=schema1)
    df1.show()
    print(df1.head(2))


def learn8_segy_to_dataframe_v1():
    cnt_single_read = 500  # 设置单次读取数，次数多了可能报错
    path2 = 'F:/Research/data/F3_entire.segy'
    with segyio.open(path2, mode='r+') as F3_entire:
        # 将segy数据的trace提取出来
        data1 = F3_entire.trace.raw[:]

        print('SEGY数据条数：', len(data1))
        cnt_for_write = len(data1) + 1  # 设置for循环的次数
        # 将segy的trace数据转换成为list数据
        list1 = []
        for i0 in range(cnt_for_write):
            for i1 in range(cnt_single_read):
                list_mini1 = []
                for i2 in range(len(data1[0])):
                    # 将数据类型强制转换成为float格式，因为DataFrame不支持numpy.float格式的数据(不支持numpy数据)。如果不转则报错。
                    # 必须是将list每一项逐项转换，不支持将list整体都转换为float格式数据
                    list_mini1.append(float(data1[i1][i2]))
                list1.append(list(list_mini1))
        # 生成DataFrame的标签schema。应为string格式。
        schema1 = []
        for i in range(len(list1[0])):
            schema1.append("{}".format(i))
        # 创建DataFrame
        df1 = ss.createDataFrame(data=list1, schema=schema1)
        df1.show()
    F3_entire.close()


# https://clowdflows.readthedocs.io/en/latest/cf_dev_wiki/example.html样例测试
def learn9():
    import urllib.request as urllib2
    import json
    somesentence = "The good the bad and the ugly"
    somesentence = somesentence.replace(" ", "+")
    url = 'http://kt.ijs.si/MartinZnidarsic/webservices/sentana/sentana.php?sentence=' + somesentence
    response = urllib2.urlopen(url).read()
    jsondata = json.loads(response)
    print("Sentiment score is: " + str(jsondata['data']['sentimentscore']))


# 将地震数据转换成为RDD和DataFrame
def learn10():
    path2 = 'F:/Research/data/F3_entire.segy'
    path3 = 'F:/Research/data/viking_small.segy'
    with segyio.open(path2, mode='r+') as F3_entire:
        # 将segy数据的trace提取出来
        data1 = F3_entire.trace.raw[:]  # raw[:2000]可以限制读取条数为2000条
        print('SEGY数据条数：', len(data1))
        # 将segy的trace数据转换成为list数据
        list1 = []
        for i in range(10):
            list_mini1 = []
            for ii in range(len(data1[0])):
                # 将数据类型强制转换成为float格式，因为DataFrame不支持numpy.float格式的数据(不支持numpy数据)。如果不转则报错。
                # 必须是将list每一项逐项转换，不支持将list整体都转换为float格式数据
                list_mini1.append(float(data1[i][ii]))
            list1.append(list(list_mini1))
        # 生成DataFrame的标签schema。应为string格式。
        schema1 = []
        for i in range(len(data1[0])):
            schema1.append(StructField('{}'.format(i), DoubleType(), True))
        print('schema1:', schema1)
        schema2 = StructType(schema1)
        # 创建RDD
        print('LIST:', list1)
        rdd1 = sc.parallelize(list1)
        print('RDD:', rdd1.collect())
        # 创建DataFrame
        print('匹配:', len(list1), len(schema2))
        df1 = ss.createDataFrame(data=list1, schema=schema2)
        df1.show()
        global global_df
        global_df = df1
        global_df.summary().show(3)
    F3_entire.close()


if __name__ == '__main__':
    learn5()
