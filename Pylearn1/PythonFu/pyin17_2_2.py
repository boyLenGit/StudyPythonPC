# 创建TCP客户端，与17_2_1组合使用
import socket
import time, random

send_data1 = '测试9879794645646'
host = '192.168.3.22'
port = 8080

def SendData(senddata):
    socket1 = socket.socket()
    socket1.connect((host, port))
    socket1.send(send_data1.encode())
    recvdata1 = socket1.recv(1024).decode()
    socket1.close()
    return recvdata1

while True:
    time.sleep(random.random())
    recvdata1 = SendData(send_data1)
    print('接收到的数据为：', recvdata1)

