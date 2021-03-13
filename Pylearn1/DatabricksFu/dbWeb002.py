from pyspark.sql import *
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

sc = SparkContext("local", "first app")
ss = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value") \
    .getOrCreate()
