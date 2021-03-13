# 创建TCP服务端，用同局域网下的随意设备的浏览器作为客户端
import socket

host = '192.168.3.22'  # 网址： 192.168.3.22:8080
port = 8080
string1=b'HTTP/1.1 200 OK\r\n\r\nHello World+{cnt}'
cnt=0
socket1 = socket.socket()
socket1.bind((host, port))
socket1.listen(5)
print('SENDING')

while True:
    acha = socket1.accept();#acha：(<socket.socket fd=584, family=AddressFamily.AF_INET, type=SocketKind.SOCK_STREAM, proto=0, laddr=('192.168.3.22', 8080), raddr=('192.168.3.3', 58825)>, ('192.168.3.3', 58825))
    conn, addr=acha #conn：<socket.socket fd=584, family=AddressFamily.AF_INET, type=SocketKind.SOCK_STREAM, proto=0, laddr=('192.168.3.22', 8080), raddr=('192.168.3.3', 58825)>     rddr：('192.168.3.3', 58825)
    data = conn.recv(1024)        #获取客户端的请求数据，最大1024字节
    print(data)
    conn.sendall(string1)
    cnt+=1
    conn.close()
