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

sc = SparkContext("local[8]", "first app")
ss = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option",
                                                                       "some-value").getOrCreate()


# 加载和转换数据模块
def learn1():

    print()


if __name__ == '__main__':
    learn1()
