# Spark&Numpy&BP+FP
import random, numpy, multiprocessing
from pyspark.sql import SparkSession
from pyspark import SparkContext

sc = SparkContext("local")
data1 = []
new_b, new_w, totalError, b_gradient, w_gradient = 0, 0, 0, 0, 0


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
    # initialization
    global new_b, new_w, totalError, b_gradient, w_gradient
    b_gradient = w_gradient = 0
    SimpleNumbers = float(len(data2))
    rdd1 = sc.parallelize(data2)
    w_current = w_starting
    b_current = b_starting

    # train
    for i1 in range(train_times):
        # print('rdd1.collect():', i1, rdd1.collect())

        # Calculate Derivative
        def train(input1):
            global new_b, new_w, b_gradient, w_gradient
            x = input1[0]
            y = input1[1]
            b_gradient += (2 / SimpleNumbers) * ((w_current * x + b_current) - y)
            w_gradient += (2 / SimpleNumbers) * ((w_current * x + b_current) - y) * x
            return input1

        # Calculate Loss
        def gradient(input2):
            #print('===', input2, end='')
            global new_b, new_w, totalError
            x2 = input2[0]
            y2 = input2[1]
            totalError += (y2 - (new_w * x2 + new_b)) ** 2
            return input2

        # Spark for train
        rdd2 = rdd1.map(lambda input1: train(input1))
        #print('')
        #print('rdd2.collect():', i1, rdd2.collect())
        #rdd2.collect()
        new_b = b_current - (lr * b_gradient)
        new_w = w_current - (lr * w_gradient)

        # Spark for gradient
        rdd3 = rdd1.map(lambda input2: gradient(input2))
        rdd3.collect()
        loss = totalError / SimpleNumbers
        print('Epoch:', i1, loss)
        if i1 % 1000 == 0:
            print('Epoch:', i1, loss)

        totalError = 0


def MainFun():
    lr = 0.001
    initial_b = 0
    initial_w = 0
    train_times = 10000
    data1 = get_data(100)
    update_gradient(data1, initial_b, initial_w, lr, train_times)


if __name__ == '__main__':
    MainFun()
