# Himmelblau函数优化实战
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d


def himmelblau(x, y):
    # himmelblau 函数实现
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


# 生成二维himmelblau函数的数据
def learn1():
    x = numpy.arange(-6, 6, 0.1)
    y = numpy.arange(-6, 6, 0.1)
    print('x,y range:', x.shape, y.shape)
    # 生成 x-y 平面采样网格点，方便可视化
    X, Y = numpy.meshgrid(x, y)
    print('X,Y maps:', X.shape, Y.shape)
    Z = himmelblau(X, Y)
    fig = plt.figure('himmelblau')
    ax = fig.gca(projection='3d')  # 必须导入Axes3D，否则该代码会报错
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    # 开始循环计算xy梯度，并更新xy梯度，从而逐渐逼近z的某个极小值
    x1 = tf.constant([-2., 2.])
    list_x0 = []
    list_x1 = []
    for step in range(200):
        with tf.GradientTape() as tape:
            tape.watch([x1])  # 加入梯度跟踪列表
            Z = himmelblau(x1[0], x1[1])
        grads = tape.gradient(Z, x1)
        x1 = x1 - 0.001 * grads
        if step % 20 == 0: print('step:', step, ' x:', x1.numpy())
        list_x0.append(x1[0])
        list_x1.append(x1[1])
    plt.plot(range(200), list_x0, list_x1)
    plt.title('XY Grad')
    plt.show()


if __name__ == '__main__':
    learn1()
