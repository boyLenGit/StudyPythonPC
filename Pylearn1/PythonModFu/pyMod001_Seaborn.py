#Seaborn绘图功能测试
import seaborn as sns
import numpy as np
import pandas
import matplotlib.pyplot as plt

x = np.arange(8)/10
y = np.array([1,5,3,6,2,4,5,6])
#常规barplot的使用
data_pandas = pandas.DataFrame({"x-轴": x,"y-轴": y})
print('df:',data_pandas)
plt.rcParams['font.family'] = ['STKaiti']
sns.barplot(x="x-轴",y="y-轴",data=data_pandas, orient='h',palette="ch:2.5,-.2,dark=.3")
plt.xticks(rotation=0)
plt.show()
#hue=的使用（图例）
ax = sns.barplot(x="x-轴",y="y-轴", hue="x-轴", data=data_pandas, dodge=False)
plt.show()