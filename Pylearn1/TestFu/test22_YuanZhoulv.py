from __future__ import print_function
import sys
from random import random
from operator import add
from pyspark.sql import SparkSession
import time

time1 = time.time()
print((int(round(time1 * 1000))))
num1 = 10000000  # MengTeKaLuoFangFa
cnt = 0
for i in range(num1):
    x, y = random(), random()
    if (x ** 2 + y ** 2) < 1:
        cnt = cnt + 1
pi = cnt * 4 / num1
print('cnt:', cnt, 'pi:', pi)
