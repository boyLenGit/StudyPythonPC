# Progress进程的使用
from multiprocessing import Process
from PythonFu import pybook13_1_DataForUSDA


def test(interval):
    pybook13_1_DataForUSDA.CreateAndInsertData_64W()


def test2(interval):
    print('我是子进程')


def main():
    print('主进程开始')
    p = Process(target=test, name='boyLenCal', args=(1,))
    p.start()
    p1 = Process(target=test2, args=(1,))
    p1.start()
    print('zhu')


if __name__ == '__main__':
    main()
    print('avc%sefe%s' % ('23', '1243'))
