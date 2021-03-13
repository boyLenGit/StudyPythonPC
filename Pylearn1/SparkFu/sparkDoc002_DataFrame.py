from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark
import pyspark.sql.functions as functions

sc = SparkContext("local", "first app")
ss = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value") \
    .master('local[4]').getOrCreate()


# DataFrame模块测试
def learn1():
    data1 = [('Alice', 1), ('Boy', 2), ('Girl', 3), ('A', 4), ('B', 5), ('C', 6)]
    data_name1 = ['name', 'age']
    df = ss.createDataFrame(data=data1, schema=data_name1)
    print('1:', df.collect())
    print('2:', ss.range(1, 10, 2).collect())
    ss2 = ss.newSession()
    df.show()


# 验证
def learn2():
    list1 = [[1., 144.5, 5.9, 33., 77.], [1., 144.5, 5.9, 33., 77.], [2., 167.2, 5.4, 45., 99.],
             [3., 124.1, 5.2, 23., 55.]]
    schema1 = ['id', 'weight', 'height', 'age', 'gender']
    df1 = ss.createDataFrame(list1, schema=schema1)
    df1.show()

    # foreach 验证
    def test1(df1):
        print('测试：', df1.id, end='  ')
        print('测试：', df1.age)

    df1.foreach(test1)
    # intersect 验证
    list2 = [[1., 144.5, 5.9, 33., 77.], [1., 144.5, 5.9, 33., 77.], [22., 167.2, 5.4, 45., 99.],
             [33., 124.1, 5.2, 23., 55.]]
    schema2 = ['id', 'weight', 'height', 'age', 'gender']
    df2 = ss.createDataFrame(data=list2, schema=schema2)
    df3 = df1.intersect(df2)
    df3.show()
    # intersectAll 验证
    df4 = df1.intersectAll(df2)
    df4.show()
    # isLocal验证
    print('isLocal验证:')
    print(df1.isLocal())
    # join验证
    print('join验证:')
    print('df1\df2:')
    df1.show()
    df2.show()
    df5_join = df1.join()
    print('df5:')
    df5_join.show()


# 验证
def learn3():
    list1 = [[10., 144.5, 5.9, 33., 77.], [10., 144.5, 5.9, 33., 73.], [2., 167.2, 5.4, 45., 99.],
             [3., 124.1, 5.2, 23., 55.]]
    schema1 = ['id', 'weight', 'height', 'age', 'gender']
    df1 = ss.createDataFrame(list1, schema=schema1)
    df1.show()
    # limit
    df1.limit(3).show()
    # orderBy
    df2 = df1.orderBy(df1.id)
    df2.show()
    df1.orderBy(df1.id, df1.gender).show()
    # rdd
    rdd1 = df1.rdd
    print(rdd1.collect())
    # randomSplit
    print('randomSplit:')
    df3 = df1.randomSplit(weights=[0.25, 0.75])
    df3[0].show()
    df3[1].show()
    # repartition  getNumPartitions
    df4 = df1.repartition(3)
    df4.show()
    print(df4.rdd.getNumPartitions())
    # replace
    print('REPLACE:')
    df5 = df1.replace(to_replace=[10.0, 5.9], value=[100.0, 50.9], subset='height')
    df5.show()
    # schema
    print(df1.schema)
    # select
    df1.select('*').show()
    # selectExpr
    df1.selectExpr('age*2', 'abs(age)').show()
    # show
    print('SHOW:')
    df1.show(n=10, truncate=2, vertical=True)
    df1.show(n=10, truncate=2, vertical=False)
    # stat
    print(df1.stat)
    # storageLevel
    print('storageLevel1：', df1.storageLevel)
    print('storageLevel2：', df1.cache().storageLevel)


def learn4():
    print('初始化：')
    list1 = [[10., 144.5, 5.9, 33., 77.], [10., 144.5, 5.9, 33., 73.], [2., 167.2, 5.4, 45., 99.],
             [3., 124.1, 5.2, 23., 55.]]
    list1_1 = [[2., 167.2, 5.4, 45., 99.], [3., 124.1, 5.2, 23., 55.], [3333., 3124.1, 335.2, 3333., 3355.]]
    schema1 = ['id', 'weight', 'height', 'age', 'gender']
    df1 = ss.createDataFrame(list1, schema=schema1)
    df1.show()
    list2 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]
    schema2 = ['id', 'weight', 'height', 'age', 'gender']
    df2 = ss.createDataFrame(list1_1, schema=schema2)
    df2.show()
    # subtract
    print('subtract：')
    df1.subtract(df2).show()
    # summary
    print('summary：')
    df1.summary().show()
    # tail
    print('tail：')
    print(df1.tail(10))
    # toDF
    print('toDF：')
    schema3 = ['A', 'S', 'D', 'F', 'G']
    df1.toDF('A', 'S', 'D', 'F', 'G').show()
    rdd1 = sc.parallelize(list1)
    rdd1.map(lambda x: x).toDF(schema3).show()
    # toJSON
    print('toJSON：')
    print(df1.toJSON())
    print(df1.toJSON().first())
    # toPandas
    print('toPandas：')
    print(df1.toPandas())
    # union
    print('union：')
    df3 = ss.createDataFrame(data=list1_1, schema=schema3)
    df4 = df1.union(df3)
    df4.show()
    # unionAll
    print('unionAll：')
    df5 = df1.unionAll(df3)
    df5.show()
    # unionByName
    print('unionByName：')
    df6 = df1.unionByName(df2)
    df6.show()
    # unpersist
    print('unpersist：')
    df6.unpersist()
    # write
    print('write：')
    a = df1.write
    print(a)
    # writeStream
    print('writeStream:')
    b = df1.writeStream
    print(b)


def learn5():
    list1_1 = [[2., 167.2, 5.4, 45., 99.], [3., 124.1, 5.2, 23., 55.], [3333., 3124.1, 335.2, 3333., 3355.]]
    schema1 = ['id', 'weight', 'height', 'age', 'gender']
    df1 = ss.createDataFrame(list1_1, schema=schema1)
    df1.show()
    df2 = df1.select('gender', functions.lit('5').alias('OOP'))
    df2.show()

# groupBy验证
def learn6():
    list1_1 = [[3., 167.2, 5.4, 45., 99.], [3., 125., 5.2, 23., 55.], [3., 124.1, 5.2, 23., 55.], [3333., 3124.1, 335.2, 3333., 3355.]]
    schema1 = ['id', 'weight', 'height', 'age', 'gender']
    df1 = ss.createDataFrame(list1_1, schema=schema1)
    df1.show()
    # df1.groupby('id').agg({'weight': 'mean'}).show()
    df1.groupby('id').agg({'weight': 'max', 'height': 'max'}).show()
    df1.groupby('id').avg().show()


if __name__ == '__main__':
    learn6()
