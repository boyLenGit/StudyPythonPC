# TensorFlow2.3 线性模型实战 ——基于numpy的梯度下降算法，求解y=wx+b的最优w,b，使y=wx+b与总样本分布的拟合度最高
import numpy, multiprocessing, time

data1 = []


# 生成一段随机采样数据
def GetSimpleData(SimpleNumber):
    for i in range(SimpleNumber):
        x1 = numpy.random.uniform(-10., 10.)  # 随机采样输入x
        eps1 = numpy.random.normal(0., 0.1)  # 采样高斯噪声:均值为0，方差为0.1^2的高斯分布N(0,0.1^2)中随机采样噪声
        y1 = 1.477 * x1 + 0.089 + eps1  # 模型输出
        data1.append([x1, y1])
    data2 = numpy.array(data1)  # 将二维列表[[1],[2],[3],[4]]转换为2D Numpy数组：[1]\n[2]\n[3]\n[4]
    return data2


# 计算数据的误差
def CalcuError(b1, w1, data2):
    totalError = 0
    for i in range(0, len(data2)):
        x2 = data2[i, 0]
        y2 = data2[i, 1]
        totalError += (y2 - (w1 * x2 + b1)) ** 2  # 这个式子就是loss函数，每个循环都累加求和
    return totalError / float(len(data2))


# 计算偏导数（计算梯度）
def CalcuDerivative(b_current, w_current, data2, lr):  # b_current、w_current为当前值
    b_gradient = w_gradient = 0  # b_gradient、w_gradient为梯度值，先初始化他们
    SimpleNumbers = float(len(data2))  # 总样本数
    for i in range(0, len(data2)):
        x = data2[i, 0]
        y = data2[i, 1]  # 取出设定的随机样本x y
        b_gradient += (2 / SimpleNumbers) * ((w_current * x + b_current) - y)
        w_gradient += (2 / SimpleNumbers) * ((w_current * x + b_current) - y) * x
    new_b = b_current - (lr * b_gradient)
    new_w = w_current - (lr * w_gradient)  # 通过加负梯度来降低b,w值。通过多次运行该函数，逐渐得到loss最小值下的b,w
    return new_b, new_w


# 更新梯度
def UpdateGradient(data2, b_starting, w_starting, lr, num_iterations):
    global loss
    b = b_starting
    w = w_starting
    for step in range(num_iterations):  # num_iterations为迭代次数
        b, w = CalcuDerivative(b, w, data2, lr)
        loss = CalcuError(b, w, data2)
        if step % 1000 == 0:
            print(f'iteration:{step}, loss:{loss}, w:{w}, b:{b}')
    return [b, w], loss


# 主训练的函数
def MainFun():
    time1_1 = time.time()
    lr = 0.001  # 学习率
    # 设定w,b的初始值，w b初始值多少不重要，因为经过数次迭代w,b都会趋于正确理论值
    initial_b = 0
    initial_w = 0
    num_iterations = 100000  # 设定训练优化次数
    data1 = GetSimpleData(100)  # 获取样本
    [b, w], losses = UpdateGradient(data1, initial_b, initial_w, lr, num_iterations)
    time1_2 = time.time()
    print(time1_2 - time1_1)
    loss = CalcuError(b, w, data1)
    print(f'Final loss:{loss}, w:{w}, b:{b}')


if __name__ == '__main__':
    Progress1 = multiprocessing.Process(target=MainFun, name='boyLenRecv')
    Progress1.start()
    Progress1.join()
    print('进程关闭')
