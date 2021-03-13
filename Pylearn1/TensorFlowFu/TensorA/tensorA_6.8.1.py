# 油耗实战研究
import tensorflow as tf
import pandas
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn

def GetData():
    dataset_path = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names =  ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin']
    dataset_raw = pandas.read_csv(dataset_path,names=column_names,na_values="?",sep=" ",comment="\t",skipinitialspace=True) #！！！要用双引号，因为数据中是双引号
    dataset_1 = dataset_raw.copy()
    dataset_1 = dataset_1.dropna()#默认删除dataframe中所有带空值的行
    dataset_only_origin = dataset_1.pop('Origin')  #将Origin列单独分裂出来，成为一个单独的pandas项，原数据中Origin列被删除
    dataset_1['USA'] = (dataset_only_origin == 1)*1.0  #如果项==1，则为True（1），True*1=1*1=1，下面逻辑相同
    dataset_1['Europe'] = (dataset_only_origin == 2)*1.0
    dataset_1['Japan'] = (dataset_only_origin == 3)*1.0
    dataset_train = dataset_1.sample(frac=0.8,random_state=0) #将数据行随机分为训练集与测试集，比例8:2。
    seaborn.pairplot(dataset_train[["Cylinders", "Displacement", "Weight", "MPG"]], diag_kind="kde")#绘图
    plt.show()  #这个plt也控制seaborn模块，如果没有这个就会和下面的plot冲突
    dataset_test = dataset_1.drop(dataset_train.index)
    labels_train = dataset_train.pop('MPG')# 将标签值从数据中提取出来，用于后面训练作为y使用
    labels_test = dataset_test.pop('MPG')
    stats_train = dataset_train.describe()  # 查看训练集的输入x的统计数据
    stats_train = stats_train.transpose()   # 将数据矩阵转置
    normdataset_train = ( (dataset_train-stats_train['mean']) / (stats_train['std']) )  #每一列减去该列的平均值，再除标准差
    normdataset_test = ( (dataset_test - stats_train['mean']) / (stats_train['std']) )
    db_train = tf.data.Dataset.from_tensor_slices((normdataset_train.values,labels_train.values))
    db_train = db_train.shuffle(100).batch(256)  #shuffle是将数据打乱，shuffle中的数值越大，打乱的程度越大
    return db_train, labels_train

class Network(keras.Model):

    def __init__(self):
        super(Network, self).__init__()
        self.dense1 = layers.Dense(64, activation='selu')
        self.dense2 = layers.Dense(48, activation='selu')
        self.dense3 = layers.Dense(32, activation='selu')
        self.dense4 = layers.Dense(16, activation='selu')
        self.dense5 = layers.Dense(1, activation='selu')

    def call(self, inputs, training=None, mask=None):
        layer_connect = self.dense1(inputs)
        layer_connect = self.dense2(layer_connect)
        layer_connect = self.dense3(layer_connect)
        layer_connect = self.dense4(layer_connect)
        layer_connect = self.dense5(layer_connect)
        return layer_connect

def TrainModel(db_train,traintimes):
    list_MSE = [];list_MAE = [] #list_MSE就是Loss
    model = Network()
    model.build(input_shape=(None,9))
    model.summary()
    optimizer1 = tf.keras.optimizers.RMSprop(0.001)#创建优化器，设置学习率
    for epoch in range(traintimes):
        for step, (x, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                out = model(x)
                loss_MSE = tf.reduce_mean(tf.losses.MSE(y,out))
                loss_MAE = tf.reduce_mean(tf.losses.MAE(y,out))
                if step % 20 == 0:print(epoch,'loss_MSE：',float(loss_MSE),'loss_MAE：',float(loss_MAE))
                grads1 = tape.gradient(loss_MSE, model.trainable_variables)
                optimizer1.apply_gradients((zip(grads1, model.trainable_variables))) #model.trainable_variables就包含了所有wb参数
        list_MSE.append(float(loss_MSE));        list_MAE.append(float(loss_MAE))
    return loss_MSE,loss_MAE,model.trainable_variables,list_MSE,list_MAE

def ControlCenter():
    traintimes = 200  # 训练次数
    db_train, labels_train = GetData()
    loss_MSE, loss_MAE, trainable_variables, list_MSE, list_MAE = TrainModel(db_train, traintimes)
    x_of_loss = [i for i in range(0, traintimes)]
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.family'] = ['STKaiti']
    plt.plot(x_of_loss, list_MAE, color='blue', label='训练')
    plt.title('神经网络层实现');    plt.xlabel('Epoch训练次数');    plt.ylabel('loss均方差')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    ControlCenter()
