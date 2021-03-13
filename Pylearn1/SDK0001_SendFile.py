#文件发送客户端
import socket,sqlite3,sys,time,os,random

host = '192.168.3.22';port = 8091
string_file='E:\\NUT_DATA_BIG.txt'
TranSpeed_Single='102400'  #传输单元大小
filesize1 = str(os.path.getsize(string_file)) #文件大小
fileName1 = os.path.basename(string_file)   #ospfji.mp3

#文件读取函数
def SendFile_TCP_Gen2():
    socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket2.connect((host, port))
    FakeList1 = filesize1+'^'+TranSpeed_Single+'^'+fileName1
    socket2.send(FakeList1.encode())
    with open(string_file,'rb') as file2:
        while True:
            StartTime=time.time()
            RecvString=socket2.recv(int(TranSpeed_Single));            print('来自服务器的信息：',RecvString.decode())
            for file_line1 in file2:
                socket2.send(file_line1)
            break
    time.sleep(random.random())   #如果不延时一小段时间，接收端会出错，会将上段信息与此段信息混合
    socket2.send(b'102455231')
    socket2.close()
    EndTime=time.time()
    CaluTime = round(EndTime - StartTime, 2)
    return CaluTime


if __name__ == '__main__':
    print('文件大小：',int(filesize1)/1000,'KB')
    CaluTime=SendFile_TCP_Gen2()
    print('传输时间：' + str(CaluTime) + 's','；速度：{:.2f}'.format( ( int(filesize1) / (CaluTime) ) /1000 ) ,'KB/s')