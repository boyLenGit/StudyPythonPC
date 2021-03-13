#  老师指导下的Spark编程验证
from pyspark.sql import SparkSession
from pyspark import SparkContext
from random import random
import numpy
import pandas as pd
import time as time1
import pandas
from scipy import stats as stats
from operator import add
import pyspark.mllib.stat as stat



def compare1():
    sc = SparkContext("local")
    ss = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value") \
        .master('local[4]').getOrCreate()
    num1 = 100
    rdd1 = sc.parallelize([[random(), random()] for i in range(num1)])
    df1 = ss.createDataFrame(data=[[random(), random()] for i1 in range(num1)], schema=['x', 'y'])
    df1.show()
    rdd1.collect()


def compare2():
    sc = SparkContext("local[12]")
    list1 = [('a', 1), ('b', 2), ('1', 3), ('d', 4), ('2', 5)]
    list2 = [('data', random()) for i1 in range(10)]
    list3 = [random() for i1 in range(100)]
    print(list3)
    print(sc.parallelize(list3).sortBy(lambda x: x).collect())


def compare3():
    print('Compare: Calculate Corr')
    print('-----------panda corr-----------')
    C, N = 100, 1000
    data = pandas.DataFrame(numpy.random.randn(C * N).reshape(C, -1))
    start = time1.time()
    res1 = data.corr(method='pearson')
    res2 = data.corr(method='spearman')
    res3 = data.corr(method='kendall')
    print(len(res1), len(res2), len(res3))
    end = time1.time()
    print("panda total cost : {}".format(end - start))

    print('-----------spark+stats corr-----------')
    sc = SparkContext("local[12]")
    data = pd.DataFrame(numpy.random.randn(C * N).reshape(C, -1))

    def pearsonr(n):
        res = [stats.pearsonr(data.iloc[:, n], data.iloc[:, i])[0] for i in range(data.shape[1])]
        return res

    def spearmanr(n):
        res = [stats.spearmanr(data.iloc[:, n], data.iloc[:, i])[0] for i in range(data.shape[1])]
        return res

    def kendalltau(n):
        res = [stats.kendalltau(data.iloc[:, n], data.iloc[:, i])[0] for i in range(data.shape[1])]
        return res

    start = time1.time()
    res1 = sc.parallelize(numpy.arange(N)).map(lambda x: pearsonr(x)).collect()
    res2 = sc.parallelize(numpy.arange(N)).map(lambda x: spearmanr(x)).collect()
    res3 = sc.parallelize(numpy.arange(N)).map(lambda x: kendalltau(x)).collect()
    end = time1.time()
    print(len(res1), len(res2), len(res3))
    print("spark+stats total cost : {}".format(end - start))


def compare4():
    print('Compare: Calculate Pi')
    sc = SparkContext("local[24]")
    print('---Calculate Pi---')
    spark = SparkSession.builder.appName("PythonPi").getOrCreate()
    num1 = 100000000  # MengTeKaLuoFangFa

    # ------Spark Offical----
    time3_1 = time1.time()
    def f(_):
        x1, y1 = random(), random()
        return 1 if ((x1 ** 2 + y1 ** 2) <= 1) else 0
    count1 = sc.parallelize(range(num1)).map(f).reduce(add)
    pi3 = (4.0 * count1 / num1)
    time3_2 = time1.time()
    time_cost3 = time3_2 - time3_1
    print('Spark2 Cost Time:', time_cost3, 's, Total Num:', num1, 'pi:', pi3)

    # ------Python------
    time2_1 = time1.time()
    cnt2 = 0
    for i in range(num1):
        x, y = random(), random()
        if (x ** 2 + y ** 2) < 1:
            cnt2 = cnt2 + 1
    pi2 = cnt2 * 4.0 / num1
    time2_2 = time1.time()
    time_cost2 = time2_2 - time2_1
    print('Python Cost Time:', time_cost2, 's, Total Num:', num1, 'pi:', pi2)


def compare5():
    print('---Compare Sort---')
    from pyspark import SparkContext
    sc = SparkContext("local")
    num1 = 10000000
    list3 = [random() for i1 in range(num1)]
    print('start')
    time1_1 = time1.time()
    print(sc.parallelize(list3).sortBy(lambda x: x).take(10))
    time1_2 = time1.time()
    list4 = list3
    list4.sort()
    time2_2 = time1.time()
    print(list4[:10])
    print ('Cost Time-->Spark:', time1_2 - time1_1, 'Python:', time2_2 - time1_2)


def compare6():
    sc = SparkContext("local[15]")
    C, N = 100, 1000
    rdd1 = sc.parallelize([numpy.random.randn(100) for i in range(1000)], 1000)
    start = time1.time()
    res1 = stat.Statistics.corr(x=rdd1, method="pearson")
    res2 = stat.Statistics.corr(x=rdd1, method="spearman")
    end = time1.time()
    print("spark+mllib total cost : {}".format(end - start), "len:{0} {1}".format(len(res1), len(res2)))


compare6()

