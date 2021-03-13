# pyspark.sql.functions功能验证
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

sc = SparkContext("local", "first app")
ss = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value") \
    .getOrCreate()


# P52 4.2熟悉你的数据——4.2.1描述性统计
def learn1():
    rdd1 = sc.textFile('F:/Research/data/ccFraud.csv')
    header = rdd1.first()
    # 使用filter删除标题行
    rdd2 = rdd1.filter(lambda row: row != header)
    # 使用map用逗号分割出每一行
    rdd2 = rdd2.map(lambda row: [int(elem) for elem in row.split(',')])
    # string.split(',')是将string根据逗号拆分为list
    list1 = header.split(',')
    print('list1:', list1)
    # 创建schema，通过单行for语句创建。
    schema1 = types.StructType([*[
        types.StructField(h[1:-1], types.IntegerType(), True) for h in list1
    ]])
    # 由rdd与schema构成df1
    df1 = ss.createDataFrame(rdd2, schema1)
    df1.show()


# groupBy测试
def learn2():
    x = sc.parallelize([1, 2, 3])
    y = x.groupBy(lambda x: 'A' if (x % 2 == 1) else 'B')
    y1 = y.collect()[0]
    for i in y.collect():
        for ii in i[1]:
            print('groupBy:', i, '----', ii)
    print('groupBy-y:', y.collect())


def learn3():
    df1 = ss.read.csv('F:/Research/data/ccFraud.csv', header=True, inferSchema=True)
    df1.show()
    # 按gender列对df进行分组，并统计每组的行数
    df2 = df1.groupby('gender').count()
    df2.show()
    df3 = df1.describe(['balance', 'numTrans', 'numIntlTrans'])
    df3.show()
    # 检查偏度
    df1.agg({'balance': 'skewness'}).show()
    df1.agg(functions.max('balance').alias('max'), functions.avg('balance').alias('avg'),
            functions.mean('balance').alias('mean'), functions.stddev('balance').alias('stddev'),
            functions.sum('balance').alias('sum'), functions.skewness('balance').alias('skewness'),
            functions.variance('balance').alias('variance'),
            functions.sumDistinct('balance').alias('sumDistinct')).show()
    corr1 = df1.corr('balance', 'numTrans')
    print(corr1)


# DataFrame可视化
def learn4():
    df1 = ss.read.csv('F:/Research/data/ccFraud.csv', header=True, inferSchema=True)
    df1.show()
    data1 = df1.select('balance').rdd.flatMap(lambda row: row)
    print('1:', data1)
    data1 = data1.histogram(5)
    print('2:', data1)
    data = {'bins': data1[0][:-1], 'freq': data1[1]}
    print('3:', data)
    # Matplotlib绘图
    plt1 = plt.figure(figsize=(12, 9))
    subplot1 = plt1.add_subplot(2, 2, 1)
    subplot1.bar(x=data['bins'], height=data['freq'], width=4000)
    subplot1.set_title('balance')

    subplot2 = plt1.add_subplot(2, 2, 4)
    subplot2.bar(x=data['bins'], height=data['freq'], width=500)
    subplot2.set_title('balance')
    plt1.show()
    # Boken绘图
    charts1 = charts.Bar(data, values='freq', label='bins', title='Histogram of \'balance\'')
    charts.show(charts1)
    # 在性别中各抽取0.02的男女数量，并将抽取数据中['balance', 'numTrans', 'numIntlTrans']三个列提取出来
    data_sample1 = df1.sampleBy('gender', {1: 0.0002, 2: 0.0002}).select(['balance', 'numTrans', 'numIntlTrans'])
    print('0.02%采样后的表：')
    data_sample1.show()
    # 绘制2D点状图
    data_multi = dict([
        (elem, data_sample1.select(elem).rdd.flatMap(lambda row: row).collect())
        for elem in ['balance', 'numTrans', 'numIntlTrans']
    ])
    print('点状图表：')
    print(len(data_multi), data_multi)
    data2 = {data_multi['balance'],data_multi['numTrans']}
    charts2 = charts.Scatter(data=data_multi, x='balance', y='numTrans')
    charts.show(charts2)
    charts2 = charts.Scatter(data=data2, x='balance', y='numTrans')
    charts.show(charts2)


# sampleBy验证
def learn5():
    list1 = [(1, 144.5, 5.9, 33, 'M'), (2, 167.2, 5.4, 45, 'M'), (3, 124.1, 5.2, 23, 'F'), (4, 144.5, 5.9, 33, 'M'),
             (5, 133.2, 5.7, 54, 'F'), (3, 124.1, 5.2, 23, 'F'), (5, 129.2, 5.3, 42, 'M'), ]
    schema1 = ['id', 'weight', 'height', 'age', 'gender']
    df1 = ss.createDataFrame(data=list1, schema=schema1)
    df1.show()
    df1.sampleBy(col='id', fractions={5: 1, 1: 0.8, 2: 0.9}).show()
    df1.sampleBy(col='gender', fractions={'M': 0.5, 'F': 0.5}).show()
    rdd1 = sc.parallelize([('1', 'Sam', 19), ('2', 'Ming', 17), ('3', 'John', 18)])
    print(rdd1.collect())
    print(rdd1.flatMap(lambda row: row).collect())
    print(df1.collect())


if __name__ == '__main__':
    learn2()
