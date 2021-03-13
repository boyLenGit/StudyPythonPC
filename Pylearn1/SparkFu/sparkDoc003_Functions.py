# pyspark.sql.functions功能验证
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import functions
from pyspark.sql import SQLContext

ss = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value") \
    .getOrCreate()


def learn1():
    df = ss.range(-5, 5, 2)
    print('1:', df.collect())
    print('2:', functions._create_function_over_column('abs'))
    # add_months与DataFrame使用验证
    df2 = ss.createDataFrame([('2015-04-08', 'Data1'), ('2015-09-08', 'Data2')], ['Data', 'Name'])
    print('3:', df2.collect())
    df3 = df2.select(add_months(df2.Data, 4))
    print('4:', df3.collect())
    df4 = df2.select(add_months(df2.Data, 4).alias('next_month'))
    print('5:', df4.collect())
    # alias验证
    df5 = ss.createDataFrame([('2015-04-08',), ('2015-09-08',)])
    print('6:', df5.collect())
    df6 = df5.alias('SUPER')
    print('7:', df6.
          collect())
    df7 = ss.createDataFrame([(1, 2), (3, 4), (5, 6), (7, 8)])
    print(df7.collect())
    df8 = SQLContext.createDataFrame(data=[('Alice', 2), ('Bob', 5)], schema=['name', 'age'])
    df8.groupBy().avg('age').collect()


if __name__ == '__main__':
    learn1()
