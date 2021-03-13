from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark

sc = SparkContext("local", "first app")
ss = SparkSession.builder \
    .master("local") \
    .appName("Word Count") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


def learn1():
    textFile1 = ss.read.text("Z:\README.MD")
    print('1:', textFile1.count())
    print('2:', textFile1.first())
    linesWithSpark1 = textFile1.filter(textFile1.value.contains("Shapefile编码"))  # contains用来返回包含()内容的所有列
    print('3:', textFile1.value.contains("Shapefile编码"))
    print('4:', linesWithSpark1.count(), linesWithSpark1.first())
    print('5:', textFile1.value)

def learn2():
    list1 = [[1, 1, 2, 3, 5, 8], [11, 1, 2, 3, 5, 8], [111, 1, 2, 3, 5, 8]]
    rdd = sc.parallelize(list1)
    schema1 = ['a', 'b', 'c', 'd', 'e', 'f']
    df1 = ss.createDataFrame(data=list1, schema=schema1)
    df1.show()
    # 满足lambda x % 2的分为一组，不满足x % 2被分为另一组
    result = rdd.collect()
    print(result)

if __name__ == '__main__':
    learn2()
