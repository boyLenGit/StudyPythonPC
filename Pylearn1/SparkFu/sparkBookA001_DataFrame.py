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

sc = SparkContext("local", "first app")
ss = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value") \
    .getOrCreate()


def learn1():
    rdd1 = sc.parallelize(({'id': '001', 'name': 'Sam', 'age': 19},
                           {'id': '002', 'name': 'Ming', 'age': 17},
                           {'id': '003', 'name': 'John', 'age': 18}))
    df1 = ss.read.json(rdd1)
    print('1:', rdd1.collect())
    print('2:', df1.collect())
    print('3:')
    df1.show(1)
    df1.show()
    print('4:', df1.take(5))
    print('5:')
    df1.printSchema()


def learn2():
    rdd1 = sc.parallelize([('1', 'Sam', 19),
                           ('2', 'Ming', 17),
                           ('3', 'John', 18)])
    list1 = [('1', 'Sam', 19), ('2', 'Ming', 17), ('3', 'John', 18)]
    schema1 = types.StructType([
        types.StructField('id', types.StringType(), True),
        types.StructField('name', types.StringType(), True),
        types.StructField("age", types.IntegerType(), True)
    ])
    df1 = ss.createDataFrame(rdd1, schema1)
    df1.createOrReplaceTempView("ABC")
    # df1.createOrReplaceGlobalTempView
    df1.show()
    df1.printSchema()
    df1.select('id').show()
    df1.select('*').show()
    df1.select('id').filter('id == 2').show()
    df1.filter('id = 2').show()
    # df1.filter('2').show() 错误示例
    print(df1.first())
    df1.filter(df1.age > 17).show()
    df1.filter("name like 'S%'").show()  # 新筛选方法


# SQL的使用
def learn3():
    rdd1 = sc.parallelize([('1', 'Sam', 19), ('2', 'Ming', 17), ('3', 'John', 18), ('4', 'Ding', 18)])
    schema1 = types.StructType([
        types.StructField('id', types.StringType(), True),
        types.StructField('name', types.StringType(), True),
        types.StructField("age", types.IntegerType(), True)
    ])
    df1 = ss.createDataFrame(rdd1, schema1)
    df1.createOrReplaceTempView("ABCDEF")
    df1.show()
    ss.sql("select count(1) from ABCDEF").show()  # SQL中调用df只能通过“表”来进行，不能直接引用df
    ss.sql("select id, age from ABCDEF").show()
    ss.sql("select id, age from ABCDEF where age = 18").show()
    ss.sql("select name,age from ABCDEF where name like 'M%'").show()


def learn4():
    csv_path = 'F:/Research/data/departuredelays.csv'
    txt_path = 'F:/Research/data/airport-codes-na.txt'
    df_airports = ss.read.csv(txt_path, header='true', inferSchema='true', sep='\t')
    df_airports.createOrReplaceTempView('Airports')
    df_fly_delay = ss.read.csv(csv_path, header='true')
    df_fly_delay.createOrReplaceTempView('FlyDelay')
    df_fly_delay.cache()  # 缓存df数据集
    df_airports.show(3)
    df_fly_delay.show(3)
    ss.sql('''
    select a.City, f.origin as Origin, sum(f.delay) as Delay, sum(f.distance) as Distance
    from Airports a join FlyDelay f on a.IATA = f.origin
    where a.State = 'WA'
    group by a.City, f.origin 
    order by sum(f.delay) desc
    ''').show()  # on a.IATA = f.origin是将两个表连接起来的核心
    # ?如果是两个表join的话，那group by时必须每个表至少要by各一项，否则就会报错，例如group by a.City, f.origin
    ss.sql("select a.State, sum(f.delay) as Delays "
           "from FlyDelay f join Airports a on a.IATA = f.origin "
           "where a.Country = 'USA' "
           "group by a.State").show()


def learn5():
    list1 = [(1, 144.5, 5.9, 33, 'M'), (2, 167.2, 5.4, 45, 'M'), (3, 124.1, 5.2, 23, 'F'), (4, 144.5, 5.9, 33, 'M'),
             (5, 133.2, 5.7, 54, 'F'), (3, 124.1, 5.2, 23, 'F'), (5, 129.2, 5.3, 42, 'M'), ]
    schema1 = ['id', 'weight', 'height', 'age', 'gender']
    df1 = ss.createDataFrame(list1, schema=schema1)
    df1_single = df1.distinct()
    df1_clean = df1.dropDuplicates()
    print('Count of rows: {0}'.format(df1.count()))
    print('Count of distinct rows: {0}'.format(df1_single.count()))
    df1_clean_sort = df1_clean.sort(df1_clean.id.desc())
    df1_clean_sort.show(3)
    df1_clean.sort('id', ascending=False).show(3)
    df1_clean.sort('id', ascending=True).show(3)
    df1_clean.sort(functions.asc('id')).show(3)
    df1_clean.sort(functions.desc('id')).show(3)


def learn6():
    list1 = [(1, 144.5, 5.9, 33, 'M'), (2, 167.2, 5.4, 45, 'M'), (3, 124.1, 5.2, 23, 'F'), (4, 144.5, 5.9, 33, 'M'),
             (5, 133.2, 5.7, 54, 'F'), (3, 124.1, 5.2, 23, 'F'), (5, 129.2, 5.3, 42, 'M'), ]
    schema1 = ['id', 'weight', 'height', 'age', 'gender']
    df1 = ss.createDataFrame(list1, schema=schema1)
    df1.show()
    df1.dropDuplicates().show()
    #  下面的lambda逻辑相当于subset=['weight', 'height', 'age', 'gender']，即排除id列，保留其他列
    df1.dropDuplicates(subset=[x for x in df1.columns if x != 'id']).show()
    print(df1.columns)
    print(df1.count())
    df1.agg(functions.countDistinct('id').alias('countDis1'),
            functions.count('id').alias('count'),
            functions.countDistinct('id', 'weight').alias('countDis2')).show()
    df1.withColumn('ABC', functions.monotonically_increasing_id()).withColumn('ACD', df1.id + 10).show()


#  未观测数据的处理
def learn7():
    df1 = ss.createDataFrame([
        (1, 143.5, 5.6, 28, 'M', 100000),
        (2, 167.2, 5.4, 45, 'M', None),
        (3, None, 5.2, None, None, None),
        (4, 144.5, 5.9, 33, 'M', None),
        (5, 133.2, 5.7, 54, 'F', None),
        (6, 124.1, 5.2, None, 'F', None),
        (7, 129.2, 5.3, 42, 'M', 76000),
    ], ['id', 'weight', 'height', 'age', 'gender', 'income'])
    df1.where('id ==3').show()
    #  用来检查每一列中缺失的观测数据百分比
    df1.agg(*[
        (1 - (functions.count(c) / functions.count('*'))).alias(c + '_missing')
        for c in df1.columns]).show()
    df1.agg(functions.count('weight').alias('test1'), functions.count('*').alias('test2')).show()
    df1.show()
    df1.dropna(thresh=6).show()
    df1.dropna(thresh=4, subset=['id', 'height', 'age', 'gender', 'income']).show()
    df1.dropna(thresh=3).show()
    df1.dropna(thresh=1).show()
    df1.dropna(thresh=1, subset=['age']).show()


def learn8():
    df1 = ss.createDataFrame([
        (1, 143.5, 5.6, 28, 'M', 100000),
        (2, 167.2, 5.4, 45, 'M', None),
        (3, None, 5.2, None, None, None),
        (4, 144.5, 5.9, 33, 'M', None),
        (5, 133.2, 5.7, 54, 'F', None),
        (6, 124.1, 5.2, None, 'F', None),
        (7, 129.2, 5.3, 42, 'M', 76000),
    ], ['id', 'weight', 'height', 'age', 'gender', 'income'])
    df1.fillna(False).show()
    df1.fillna(value=1000, subset=['id', 'weight', 'height', 'age']).show()
    means = df1.agg(*[functions.mean(c).alias(c) for c in df1.columns if c != 'gender']).toPandas().to_dict('records')[
        0]
    print(means)
    means['gender'] = 'missing'
    print(means)


#  测试验证agg
def learn9():
    df1 = ss.createDataFrame([
        (1, 143.5, 5.6, 28, 'M', 100000),
        (2, 167.2, 5.4, 45, 'M', None),
        (3, None, 5.2, None, None, None),
        (4, 144.5, 5.9, 33, 'M', None),
        (5, 133.2, 5.7, 54, 'F', None),
        (6, 124.1, 5.2, None, 'F', None),
        (7, 129.2, 5.3, 42, 'M', 76000),
    ], ['id', 'weight', 'height', 'age', 'gender', 'income'])
    df1.where('id ==3').show()
    #  用来检查每一列中缺失的观测数据百分比
    df1.agg(*[
        (1 - (functions.count(c) / functions.count('*'))).alias(c + '_missing')
        for c in df1.columns]).show()
    df1.agg(functions.count('weight').alias('test1'), functions.count('*').alias('test2')).show()


#  验证分位数生成函数approxQuantile
def learn10():
    df1 = ss.createDataFrame([
        (1, 143.5, 5.3, 28),
        (2, 154.2, 5.5, 45),
        (3, 342.3, 5.1, 99),
        (4, 144.5, 5.5, 33),
        (5, 133.2, 5.4, 54),
        (6, 124.1, 5.1, 21),
        (7, 129.2, 5.3, 42),
    ], ['id', 'weight', 'height', 'age'])
    cols = ['weight', 'height', 'age']
    bounds = {}
    for col in cols:
        quantiles = df1.approxQuantile(col, [0.25, 0.75], 0.05)
        print('quantiles:', quantiles)
        IQR = quantiles[1] - quantiles[0]
        bounds[col] = [quantiles[0] - 1.5 * IQR, quantiles[1] + 1.5 * IQR]
    df1.show()
    print('bounds:', bounds)
    #  用来标记df1中每个值是否是离群值，即该值是否超出所计算的范围bounds
    outliers = df1.select(
        *['id'] + [((df1[c] < bounds[c][0]) | (df1[c] > bounds[c][1])).alias(c + '_A') for c in cols])
    outliers.show()
    #  提取离群值
    df1 = df1.join(outliers, on='id')
    df1.show()
    df1.filter('weight_A = true').select('id', 'weight').show()
    df1.filter('age_A').select('id', 'age').show()


if __name__ == '__main__':
    learn10()
