# 计算均方差MSE
import tensorflow as tf;import numpy

# 标量
def learn1():
    out1 = tf.random.uniform([4, 10]);    print(out1)  # 定义4组随机数模拟网络输出，作为样本
    x1 = tf.constant([2, 3, 2, 0])  # 定义样本标签
    x1 = tf.one_hot(x1, depth=10);    print(x1)  # 将样本标签转换为独特编码
    loss1 = tf.keras.losses.mse(x1, out1);    print(loss1)  # 计算每组样本的MSE
    loss1 = tf.reduce_mean(loss1);    print(loss1)  # 计算所有组的MSE的平均值
    print(tf.constant([100, 3, 2, 0]))
    loss2 = tf.reduce_mean(tf.constant([2, 3, 2, 0]));    print(loss2)  # 计算所有组的MSE的平均值

# 向量
def learn2():
    a1 = tf.random.normal([4, 2]);    print(a1)
    a2 = tf.constant([0., 10.]);    print(a2)   #!!!如果数字后不加.则会报错，因为10是整数，10.是浮点，类型不一样;或者不加.加上dtype=tf.float32也行
    a3 = a1 + a2;    print(a3)  # 向量相加

# 矩阵
def learn3():
    a0 = tf.random.normal([2, 4])
    a1 = tf.random.normal([4, 3]);    print(a0@a1)
    a2 = tf.constant([10,0,10],dtype=tf.float32);    print(a2)
    a3 = a0@a1 + a2;    print(a3)  # 矩阵相乘就相当于是一个网络层了

# 索引的用法
def learn4():
    a1 = tf.random.normal([4,32,32,3])#模拟图片的数据，定义4维张量
    print('第1张图片数据:',a1[0])  # 调用索引获取第1张图片数据，依次为行-列-像素-颜色强度值
    print('第1张图片 第2行数据:',a1[0][1]) #调用索引获取第1张图片 第2行数据，依次为列-像素-颜色强度值
    print('第1张图片 第2行 第3列的像素:',a1[0][1][2]) #调用索引获取第1张图片 第2行 第3列的像素，依次为像素-颜色强度值
    print('第1张图片 第2行 第3列 B通道的颜色强度值:',a1[0][1][2][1]) #调用索引获取第1张图片 第2行 第3列 B通道的颜色强度值
    print('第1张图片 第2行 第3列 B通道的颜色强度值:', a1[0,1,2,1])  # 第二种索引方式

# reshape的用法、增减维度、调整维度顺序
def learn5():
    a1 = tf.range(96)
    a2 = tf.reshape(a1,[2,4,4,3]);    print(a2)
    a3 = tf.reshape(a2,[2,-1]);    print(a3) #更改张量的视图
    a4 = tf.expand_dims(a3,axis=2);    print(tf.shape(a4))#增加维度
    a5 = tf.squeeze(a4, axis=2);    print(tf.shape(a5)) #减维度
    a6 = tf.transpose(a2,perm=[1,3,2,0]);    print(tf.shape(a2),'---调整后：',tf.shape(a6)) #调整维度顺序

#复制新维度
def learn6():
    a1 = tf.constant([1,2,3])#shape=(2,)
    a2 = tf.expand_dims(a1,axis=0);print(a1,'---',a2)#shape=(1, 2);
    a3 = tf.tile(a2,multiples=[10,10]);    print('维度：',tf.shape(a3))  #如果规模在*10，就开始闪退了
    a33 = tf.random.normal([3,4])
    a4 = tf.broadcast_to(a33,[2,1,3,4]);    print(a4,'---',tf.shape(a4))

if __name__ == '__main__':
    learn6()
