# Spark&Numpy&BP+FP
import random, numpy, multiprocessing
from pyspark.sql import SparkSession
from pyspark import SparkContext

sc = SparkContext("local")
data1 = []


# Get Data
def get_data(SimpleNumber=100):
    for i in range(SimpleNumber):
        x1 = numpy.random.uniform(-10., 10.)
        eps1 = numpy.random.normal(0., 0.1)
        y1 = 1.477 * x1 + 0.089 + eps1
        data1.append([x1, y1])
    data2 = numpy.array(data1)
    return data2


def update_gradient(data2, b_starting, w_starting, lr, train_times):
    print('TEST')
    SimpleNumbers = float(len(data2))

    def train_and_gradient(x, b_starting, w_starting, train_times, SimpleNumbers):
        print('Test1', x)
        w_current = w_starting
        b_current = b_starting
        # train
        for i1 in range(train_times):
            b_gradient = w_gradient = 0
            # Calculate Derivative
            for i in range(0, len(data2)):
                x = data2[i, 0]
                y = data2[i, 1]  # 取出设定的随机样本x y
                b_gradient += (2 / SimpleNumbers) * ((w_current * x + b_current) - y)
                w_gradient += (2 / SimpleNumbers) * ((w_current * x + b_current) - y) * x
            new_b = b_current - (lr * b_gradient)
            new_w = w_current - (lr * w_gradient)
            # Calculate Loss
            totalError = 0
            for i in range(0, len(data2)):
                x2 = data2[i, 0]
                y2 = data2[i, 1]
                totalError += (y2 - (new_w * x2 + new_b)) ** 2
            loss = totalError / float(len(data2))
            if i1 % 1000 == 0:
                print('Epoch:', i1, loss)

    rdd1 = sc.parallelize(data2)
    rdd2 = rdd1.map(lambda x: train_and_gradient(x, b_starting, w_starting, train_times, SimpleNumbers))
    rdd2.collect()


def MainFun():
    lr = 0.001
    initial_b = 0
    initial_w = 0
    train_times = 10000
    data1 = get_data(100)
    update_gradient(data1, initial_b, initial_w, lr, train_times)


if __name__ == '__main__':
    MainFun()
