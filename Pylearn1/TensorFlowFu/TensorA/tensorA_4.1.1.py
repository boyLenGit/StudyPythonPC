import tensorflow as tf;import numpy


def learn1():
    # 定义常量
    a1 = 1.2  # 创建python常数
    a2 = tf.constant(1.2)  # 创建Tensorflow标量：单个数值、维度=0
    print('学习标量是如何在Tensorflow中创建：', type(a1), type(a2), tf.is_tensor(a1), tf.is_tensor(a2))
    a3 = tf.constant([1, 2., 3.3])
    print('打印张量的相关信息：', a3)
    a4 = a3.numpy()
    print('用numpy()方法将Tensorflow张量变为numpy.array数组，方便导出其他模块：', a4)
    # 定义向量
    a5 = tf.constant([1.2])  # Tensorflow向量的定义，括号内必须是list格式。要与上面的标量定义相区分
    print('创建Tensorflow向量', a5, '---', a5.shape)
    a6 = tf.constant([1.2, 5., 32., 3.13])  # Tensorflow向量的定义，括号内必须是list格式。要与上面的标量定义相区分
    print('创建2个元素的Tensorflow向量', a6, '---', a6.shape)
    print('创建2个元素的Tensorflow向量', tf.constant([1.2, 5., 32., 23643]), '---',
          tf.constant([1.2, 5., 32., 233]).shape)  # 小数部分精度最小到7位 整数位数无限大
    # 定义矩阵
    a7 = tf.constant([[1, 2], [3, 4], [5, 6]])
    print('创建Tensorflow矩阵:', a7)
    # 定义张量
    a8 = tf.constant([[[1, 2], [3, 4], [5, 6]]])  # shape=(1, 3, 2)共有三维，维度是用户定义的
    print('创建Tensorflow张量：', a8, '\n numpy格式：', a8.numpy())
    # 定义字符串
    a9 = tf.constant('Hello')
    print('创建Tensorflow字符串：', a9, tf.strings.lower(a9))
    # 定义布尔
    a10 = tf.constant(True)
    print('创建Tensorflow布尔：', a10)
    a11 = tf.constant([True, False, True])
    print('创建Tensorflow布尔：', a11)
    # 数值精度
    print('数值精度测试：', tf.constant(123456789, dtype=tf.int8), tf.constant(123456789, dtype=tf.int16),
          tf.constant(123456789, dtype=tf.int32))
    print('数值精度测试：', tf.constant(numpy.pi, dtype=tf.float16), tf.constant(numpy.pi, dtype=tf.float32),
          tf.constant(numpy.pi, dtype=tf.float64))
    # 读取精度
    a12 = tf.constant(numpy.pi, dtype=tf.float16)
    a13 = tf.constant(numpy.pi, dtype=tf.float64)
    print('读取精度：', a13.dtype, '---', a12.dtype)
    # Tensorflow精度的if写法
    if a13.dtype == tf.float64: print('Tensorflow精度的if写法', tf.float64)
    # 转换精度
    a14 = tf.cast(a13, tf.float32)
    a15 = tf.cast(a12, tf.float64)  # 低精度转高精度，数值会出问题
    print('转换精度：', a14, '---', a15)
    # 类型转换
    print('类型转换-整数与双精度：', a13, '---', tf.cast(a13, tf.double), '---', tf.cast(a13, tf.int8))
    print('类型转换-布尔与整数：', tf.cast([True, False, True], tf.int32))


def learn2():
    # 将普通张量转化为待优化张量
    a1 = tf.constant([-1, 1, 2, 1.1])
    a2 = tf.Variable(a1)
    print('普通张量：', a1, '待优化张量:', a2, '---', a2.name, '---', a2.trainable)
    a3 = tf.Variable([[1, 2], [3, 4.1]])
    print('创建优化过的张量：', a3)


def learn4_4():
    # 从Numpy.Array或Python.list创建向量
    a1 = [1, 2, 3.3];
    a2 = numpy.array([1, 2, 3.3])
    a3 = tf.convert_to_tensor(a1);
    a4 = tf.convert_to_tensor(a2)
    print('从Numpy.Array或Python.list创建向量：', a3, '---', a4)
    # 创建全0或全1的标量
    print('创建全0或全1的张量:', tf.zeros([]), '---', tf.ones([]))
    # 创建全0或全1的向量
    print('创建全0或全1的向量:', tf.zeros([1]), '---', tf.ones([1]))
    # 创建全0或全1的矩阵
    print('创建全0或全1的矩阵:', tf.zeros([2, 3]), '---', tf.ones([2, 3]))
    # 创建与a的形状一样的全0或全1矩阵
    a5 = tf.zeros([2, 3]);    a6 = tf.ones([2, 3])
    print('创建与a的形状一样的全0或全1矩阵:', tf.zeros_like(a5), tf.ones_like(a6))
    # 创建自定义数值张量
    print('创建自定义数值张量-标量：', tf.fill([], 2.3))
    print('创建自定义数值张量-向量：', tf.fill([1], 2.3))
    print('创建自定义数值张量-矩阵：', tf.fill([2, 3], 2.3))
    # 创建正太分布的张量
    print('创建正太分布的张量（随机）-均值0标准差1：', tf.random.normal([2, 3]))
    print('创建正太分布的张量（随机）-均值1标准差2：', tf.random.normal([2, 3], mean=1, stddev=2))
    # 创建均匀分布的张量
    print('创建均匀分布的张量（随机）-shape为[2,3]其余默认（默认区间[0,1]）：', tf.random.uniform([2, 3]))
    print('创建均匀分布的张量（随机）-shape为[2,3]、区间[0,10]：', tf.random.uniform([2, 3], maxval=10))
    print('创建均匀分布的张量（随机）-shape为[2,3]、区间[0,10]的整数张量）：', tf.random.uniform([2, 3], maxval=10, dtype=tf.int64))
    #创建序列
    print('创建序列：',tf.range(10))

if __name__ == '__main__':
    # learn1()
    # learn2()
    learn4_4()
    pass
