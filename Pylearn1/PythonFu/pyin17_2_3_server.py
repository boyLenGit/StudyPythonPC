#聊天软件服务端
import socket

host = '192.168.3.22'
port = 8090

socket1=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
socket1.bind((host,port))
socket1.listen(1)
ache=socket1.accept()   #设定连接方式是被动连接，等待主动方发来信息
sock1,addr1=ache
print('Server Connected')
info1=sock1.recv(1024).decode()
info1=info1.split('^')
while 'byebye' not in info1:
    if info1:
        print('接收到的内容:',info1)
        senddata=input('输入要发送的内容：')
        sock1.sendall(senddata.encode())
    if senddata=='byebye':
        break
    info1 = sock1.recv(1024).decode()   #再次接收对方的数据，并解码，用作下一次循环使用。

sock1.close() #关闭客户端套接字
socket1.close()  #关闭服务端套接字（本端）