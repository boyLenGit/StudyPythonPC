import numpy as np
import matplotlib.pyplot as plt


n = 5
# 做点
x = np.linspace(-10, 10, n)
y = np.linspace(-10, 10, n)
# 构造点
X, Y = np.meshgrid(x, y)
Z = np.tan(X + Y)
# 作图
plt.pcolormesh(X, Y, Z)
plt.show()
print('x:', x, '\ny:', y, 'X:', X, '\nY:', Y, '\nZ:', Z)
