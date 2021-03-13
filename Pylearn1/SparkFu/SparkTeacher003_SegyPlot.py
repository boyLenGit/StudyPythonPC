import matplotlib
import segyio
import matplotlib.pyplot as plt


def learn6():
    path2 = 'E:/Research/data/F3_entire.segy'
    path3 = 'E:/Research/data/viking_small.segy'
    with segyio.open(path2, mode='r+') as F3_entire:
        # 将segy数据的trace提取出来
        data1 = F3_entire.trace.raw[:]  # raw[:2000]可以限制读取条数为2000条
        print('SEGY数据条数：', len(data1))
        # 将segy的trace数据转换成为list数据
        print(data1)
    F3_entire.close()
    return data1

data1 = learn6()

# list1 = [1, 2, 1, 2, 2, 3, 4, 5, 6, 3, 2, 4, 6, 3, 2, 1]
list_x = []
for i in range(len(data1[0])):
    list_x.append(i)
# plt.plot(list_x, list1, color='red')
cnt1 = 0
for i1 in range(30):
    for i2 in range(len(data1[0])):
        data1[i1][i2] += 30000*i1
    plt.plot(data1[i1], list_x, color='red')
plt.savefig('D:/boyLen/Python/Pylearn1/SparkFu/images/SegyPlot.png')




