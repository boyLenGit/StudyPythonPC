#合并与分割章节
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def learn1():
    a1 = tf.random.normal([4,35,8])
    a2 = tf.random.normal([6,35,8])
    a3 = tf.random.normal([6,35,10])
    print(tf.shape(tf.concat([a1,a2],axis=0))) #合并第1维度
    print(tf.shape(tf.concat([a2,a3],axis=2))) #合并第3维度
    print(tf.shape(tf.concat([a1,a1],axis=0))) #合并自己

def learn2():
    #张量比较
    a1 = tf.constant([[2, 4], [6, 8]])
    a2 = tf.constant([[2, 4], [6, 10]])
    print(tf.equal(a1,a2))
    print(tf.math.equal(a1, a2))
    #数据填充
    print(tf.pad(a1, [[0,1],[2,0]]))
    a3 = tf.random.normal([4,8,8,1],mean=1.0)
    print(a3)
    print(tf.pad(a3,[[0,0],[2,2],[2,2],[0,0]]))
    print(tf.shape(tf.pad(a3,[[0,0],[2,2],[2,2],[0,0]])))

def learn3():
    a1 = tf.range(10)
    print(tf.clip_by_value(a1,3,5))  #张量数值限幅
    a1 = tf.constant([[2, 4], [6, 8], [3, 5]])
    print(a1[1])
    print(tf.gather(a1,[0,2],axis=0))   #提取维度
    a3 = tf.constant([[True,False],[True,False]])
    print(tf.where(a3))

def learn4():
    print(numpy.sinc(1)) #sinc函数
    a1 = tf.constant([1,2,3,4,5,6,7,8])
    a2 = tf.constant([11,22,33,44,55,66,77,88])
    print(tf.meshgrid(a1,a2))

def learn5():
    x = tf.linspace(-8., 8, 100)  # 设置 x 坐标的间隔
    y = tf.linspace(-8., 8, 100)  # 设置 y 坐标的间隔
    x, y = tf.meshgrid(x, y)  # 生成网格点，并拆分后返回
    print(x.shape,y.shape)
    z = tf.sqrt(x ** 2 + y ** 2)
    z = tf.sin(z) / z  # sinc 函数实现
    fig = plt.figure()
    ax = Axes3D(fig)
    # 根据网格点绘制 sinc 函数 3D 曲面
    ax.contour3D(x.numpy(), y.numpy(), z.numpy(), 50)
    plt.show()



if __name__ == '__main__':
    learn5()