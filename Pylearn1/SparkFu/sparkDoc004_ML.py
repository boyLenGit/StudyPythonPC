# pyspark.sql.functions功能验证
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import functions
from pyspark.sql import SQLContext

ss = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value") \
    .getOrCreate()


# 计算相关系数矩阵
def learn1():
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.stat import Correlation

    data = [(Vectors.sparse(4, [(0, 1.0), (3, -2.0)]),),
            (Vectors.dense([4.0, 5.0, 0.0, 3.0]),),
            (Vectors.dense([6.0, 7.0, 0.0, 8.0]),),
            (Vectors.sparse(4, [(0, 9.0), (3, 1.0)]),)]
    df = ss.createDataFrame(data, ["features"])

    r1 = Correlation.corr(df, "features").head()
    print("Pearson correlation matrix:\n" + str(r1[0]))

    r2 = Correlation.corr(df, "features", "spearman").head()
    print("Spearman correlation matrix:\n" + str(r2[0]))


if __name__ == '__main__':
    learn1()
