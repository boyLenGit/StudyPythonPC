# GraphFrame功能验证
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import functions
from pyspark.sql import SQLContext
import pyspark
import databricks
import pyspark.sql as sql
import matplotlib.pyplot as plt
from bokeh import charts
from bokeh.io import output_notebook
import graphframes
import pyspark.sql.types as typ
import pyspark.mllib.stat as stat
import numpy

sc = SparkContext("local", "first app")
ss = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option",
                                                                       "some-value").getOrCreate()


def learn0():
    path1 = 'E:/Research/data/SparkML/train.tsv'
    rdd1 = sc.textFile(path1).collect()
    df1 = ss.createDataFrame(data=rdd1, schema=None)
    df1.show()
    rdd1 = df1.rdd.map(lambda x: [i for i in x])


# 加载和转换数据模块
def learn1():
    path1 = 'E:/Research/data/SparkML/train.tsv'
    rdd1 = sc.textFile(path1)
    list1 = rdd1.collect()
    header1 = list1[0]
    print('header1：', header1)
    print('content1：', list1[1])
    rdd2 = sc.parallelize(c=list1[1:], numSlices=100)
    print('rdd2.count():', rdd2.count())
    # 替换
    rdd2.map(lambda x: x.replace())



if __name__ == '__main__':
    learn1()
