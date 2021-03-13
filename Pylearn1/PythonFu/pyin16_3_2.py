#多进程之间的通信
from multiprocessing import Queue, Process
import time
def write_task(q):
    if not q.full():
        for i in range(5):
            message1='消息'+str(i)
            q.put(message1)
            print('1:'%message1)
if __name__=='__main__':
    q=Queue(3)  #最多可以接收3条信息
    q.put('1')
    q.put('2')
    print(q.full())
    q.put('3')
    print(q.full())    #返回Ture因为消息已经3条填满了
