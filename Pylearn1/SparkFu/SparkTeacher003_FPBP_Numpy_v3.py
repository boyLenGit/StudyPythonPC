# Spark&Numpy&BP+FP
import random, numpy, multiprocessing
from pyspark.sql import SparkSession
from pyspark import SparkContext
import time

sc = SparkContext("local[8]")
data1 = []
b_current, w_current = 0, 0


# Get Data
def get_data(SimpleNumber=100):
    for i in range(SimpleNumber):
        x1 = numpy.random.uniform(-10., 10.)
        eps1 = numpy.random.normal(0., 0.1)
        y1 = 1.477 * x1 + 0.089 + eps1
        data1.append([x1, y1])
    data2 = numpy.array(data1)
    return data2


def update_gradient(data2, lr, train_times):
    SimpleNumbers = float(len(data2))
    i1 = 0

    def train_and_gradient(i1, SimpleNumbers, data2):
        # train
        global b_current, w_current
        b_gradient = w_gradient = 0

        # Calculate Derivative
        for i in range(0, len(data2)):
            x, y = data2[i, 0], data2[i, 1]  # 取出设定的随机样本x y
            b_gradient += (2 / SimpleNumbers) * ((w_current * x + b_current) - y)
            w_gradient += (2 / SimpleNumbers) * ((w_current * x + b_current) - y) * x
        new_b = b_current - (lr * b_gradient)
        new_w = w_current - (lr * w_gradient)

        # Calculate Loss
        totalError = 0
        for i in range(0, len(data2)):
            x2, y2 = data2[i, 0], data2[i, 1]
            totalError += (y2 - (new_w * x2 + new_b)) ** 2
        loss = totalError / SimpleNumbers
        b_current, w_current = new_b, new_w
        if i1 % 1000 == 0:
            print('Epoch:', i1, loss)

    rdd1 = sc.parallelize(range(train_times))
    rdd2 = rdd1.map(lambda x: train_and_gradient(x, SimpleNumbers, data2))
    rdd2.collect()


def MainFun():
    time1_1 = time.time()
    lr = 0.001
    train_times = 100000
    data1 = get_data(100)
    update_gradient(data1, lr, train_times)
    time1_2 = time.time()
    print(time1_2-time1_1)


if __name__ == '__main__':
    MainFun()
