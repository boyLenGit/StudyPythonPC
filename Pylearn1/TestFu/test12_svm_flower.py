# SVM实例，基于sklearn分析鸢尾花的二维特征，三分类问题

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score      # 也可直接调用accuracy_score方法计算准确率


# define converts(字典)
def Iris_label(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


# 1.读取数据集
path = '/dataLen/Iris.data'
# 括号中converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label(number)
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: Iris_label})  # data为一组二维数据矩阵，包含多个数据特征，每列都是1个特征

# 2.划分数据与标签
x, y = np.split(data, indices_or_sections=(4,), axis=1)  # x为数据，y为标签。数据为鸢尾花的数据特征
x = x[:, 2:4]   # 为便于后边画图显示，只选取前两维度。若不用画图，可选取前四列x[:,0:4]
train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.6, test_size=0.4)

# 3.训练svm分类器
classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')  # ovr:一对多策略
classifier.fit(train_data, train_label.flatten())  # flatten()函数用于将多维数据拍扁  fit是将数据进行训练,fit后模型就已经训练好了

# 4.计算svc分类器的准确率
print("训练集：", classifier.score(train_data, train_label))
print("测试集：", classifier.score(test_data, test_label))

tra_label = classifier.predict(train_data)  # 训练集的预测标签，也就是train_data重走一遍模型，计算预测值，为了后面计算train精度。
tes_label = classifier.predict(test_data)  # 测试集的预测标签，为了计算test精度。
print("训练集：", accuracy_score(train_label, tra_label))
print("测试集：", accuracy_score(test_label, tes_label))

# 查看决策函数
print('train_decision_function:\n', classifier.decision_function(train_data))  # (90,3)

# 5.绘制图形
# 确定坐标轴范围
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0维特征的范围内的最大、最小值,用于设定绘图的边界
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1维特征的范围内的最大、最小值
x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # 生成网络采样点
# 将生成的网络的每个点的xy坐标组成[x,y]格式，为了后面进行predict生成可视化类别分布图，因为模型
# 的train_data就是[x,y]的格式.
grid_test = np.stack((x1.flat, x2.flat), axis=1)
# 指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 设置颜色
cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])

grid_hat = classifier.predict(grid_test)  # 预测分类值，为了生成右侧的可视化类别分布图
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
print(grid_hat)
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 绘制分类图，grid_hat实际上就是经过模型预测出的标签值
print('pcolormesh:', x1.shape, x2.shape, grid_hat.shape)
plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=30, cmap=cm_dark)  # x是所有的样本点，根据标签y来控制每个样本点的颜色，绘制点图
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label[:, 0], s=30, edgecolors='k', zorder=2, cmap=cm_dark)  # 圈中测试集样本点
plt.xlabel('花萼长度', fontsize=13)
plt.ylabel('花萼宽度', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('鸢尾花SVM二特征分类')
plt.show()
