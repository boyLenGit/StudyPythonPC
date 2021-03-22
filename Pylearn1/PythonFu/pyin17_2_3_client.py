# 聊天软件客户端
import socket

socket2 = socket.socket()
host = '192.168.3.42'
# host = '180.201.161.93'
port = 8090
socket2.connect((host, port))
print('Client Connected')
info = ''
while info != 'byebye':
    senddata = input('输入发送内容：')  # input遇到回车或换行符\n便会结束
    Fakelist = host + '^' + senddata
    socket2.sendall(Fakelist.encode())
    if senddata == 'byebye':
        break
    info = socket2.recv(1024).decode()
    print('接收到的内容：', info)
socket2.close()
