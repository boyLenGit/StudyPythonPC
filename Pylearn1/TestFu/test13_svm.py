# SVM分割超平面的绘制与SVC.decision_function( )的功能
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
"""
=========================================
SVM: Maximum margin separating hyperplane
=========================================

Plot the maximum margin separating hyperplane within a two-class
separable dataset using a Support Vector Machine classifier with
linear kernel.
"""
print(__doc__)
np.random.seed(0)   # 设定随即种子
X = np.array([[3, 3], [4, 3], [1, 1]])
Y = np.array([1, 1, -1])    # 构件二分类标签。PS：二分类三分类就是看y的内容

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# get the separating hyperplane
w = clf.coef_[0]    # 返回目标函数的除w0之外的项
a = -w[0] / w[1]
xx = np.linspace(-5, 5)     # 生成50个-5~5之间的等距点
yy = a * xx - (clf.intercept_[0]) / w[1]   # 返回目标函数的w0项
print('clf.intercept_[0]:', clf.intercept_, ' w:', clf.coef_)

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], c='g')
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, alpha=0.8)   # c=Y是控制点的颜色   X[:, 0], X[:, 1]是将点的xy轴坐标分离出来，便于绘图
plt.axis('tight')
plt.show()
print('decision_function:', clf.decision_function(X), '  y:', Y)
