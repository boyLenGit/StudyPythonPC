# 线程之间通过队列QUEUE(queue)交流
from queue import Queue
import random, threading, time


class Producer(threading.Thread):
    def __init__(self, name, queue):  # 如果不命名name则默认按照Thread-N来命名
        threading.Thread.__init__(self, name=name)  # 继承Thread？
        self.data = queue

    def run(self):
        for i in range(5):
            print('%dMake:%s', (self.getName(), i))
            self.data.put(i)
            time.sleep(random.random())
        print('%sMake Done', self.getName())


class Consumer(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.data = queue

    def run(self):
        for i in range(5):
            val1 = self.data.get()
            print('%dTake:%s', (self.getName(), val1))
            time.sleep(random.random())
        print('%sTake Done', self.getName())


if __name__ == '__main__':
    print('START')
    queue = Queue()
    producer = Producer('Producer', queue)
    consumer = Consumer('Consumer', queue)
    producer.start();consumer.start();
    # producer.join();consumer.join()
