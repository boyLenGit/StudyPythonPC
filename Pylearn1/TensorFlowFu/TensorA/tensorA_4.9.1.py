#Tensorflow数学运算
import tensorflow as tf;import numpy,os

#加减乘除
def learn1():
    a1 = tf.range(5)
    a11 = tf.cast(a1, dtype=tf.float32)
    a2 = tf.constant(2)
    print("加减乘除：",a1+a2,a1-a2,a1*a2,a1/a2)
    print('除：',a1/a2,'整除：',a1//a2,'求余：',a1%a2) #除、整除、求余
    print('乘方：',tf.pow(a1,a2),'；或：',a1**a2) #乘方，必须是同类数int或float，int*float报错
    print('开根：', tf.pow(a11, 0.5), '；或：', a11 ** (0.5))  # 开根  开根对象必须是float格式，不能是int格式
    print('开根2：',tf.sqrt(a11))  # 开根  开根对象必须是float格式，不能是int格式
    print('开方：',tf.square(a11))  # 开方
    print('指数1：',a1**4,'指数2：',tf.pow(a1,a2),'指数e：',tf.exp(a11))  #指数非开根的可以是int，exp必须是float
    print('对数log：',tf.math.log(3.))  #必须是float，且默认是对e求对数
    print('对数log2(4)：',(tf.math.log(4.))/(tf.math.log(2.)))

#矩阵相乘
def learn2():
    a1 = tf.random.normal([50,100,100,99])
    a2 = tf.random.normal([50,100,99,100]) #[99,100,100,99会显存溢出报错]
    a3 = a1@a2;    print('矩阵相乘：',tf.shape(a3))
    a4 = tf.random.normal([99,98])
    a5 = tf.matmul(tf.random.normal([100,100,99]),a4);    print('矩阵相乘2：',tf.shape(a5))

if __name__ == '__main__':
    learn1()