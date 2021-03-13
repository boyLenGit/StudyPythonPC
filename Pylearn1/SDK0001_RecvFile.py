# 文件接收服务端。支持任何格式任何大小的文件，多进程接收大规模数据文件、自动根据单行数据规模生成SQLite创建与存储指令，将接收的数据文件处理后存储到SQLite数据库中，具有ControlCenter函数  【作为服务端要先启动】
import socket, multiprocessing, sqlite3, time, random

host = '192.168.3.42'
port = 8091  # IP必须是本机的IP地址，客户端要跟本机IP地址一致才可。
string_StoreFilePath_First = 'G:\\'
TranSpeed_Single = 1024000


def TextCuter2(list2):
    listTextCuter1 = []  # 如果是全局变量的话，listTextCuter1的内容会一直增大，这里用作每次循环清零1次
    for stringtextCuter1 in list2:
        listTextCuter1.append(stringtextCuter1.strip('~').strip('\n'))
    return listTextCuter1


def ModFun_CreateAndInsertData(tablename, string_file_Full):  # 传递的是 子数据库名,已下载文件的路径全称
    cnt1 = 1  # 记录数据条数，同时作为数据库的ID序号
    string_Create_Full, string_Insert_Full, LenOfSQL = ModFun_SmartBuildSQLiteExecute(string_file_Full)
    connection1 = sqlite3.connect('food.db')
    cursor1 = connection1.cursor()
    try:
        cursor1.execute(string_Create_Full.format(tablename2=tablename))
    except sqlite3.OperationalError:
        print('数据表 {tablename2} 已经存在！'.format(tablename2=tablename))
    try:
        query = string_Insert_Full.format(tablename2=tablename)  # 导入数据库用
        for line in open(string_file_Full):
            fields = line.split('^')
            vals = TextCuter2(fields)  # print('vals:',vals)
            list_21W = [str(cnt1)]
            cnt1 += 1
            list_21W.extend(vals)
            cursor1.execute(query, list_21W)
        print(str(cnt1), 'x', LenOfSQL, '条数据已写入数据库，最后一条为：', str(list_21W))
    except sqlite3.IntegrityError:
        print('数据表 {tablename2} 已经写入数据！错误码：UNIQUE constraint failed: NUT2.id'.format(tablename2=tablename))
    connection1.commit()
    connection1.close()


# 根据数据文件txt的样式，自动生成SQLite指令
def ModFun_SmartBuildSQLiteExecute(string_file):
    with open(string_file, 'r') as file1:
        string_line = file1.readline()
        file1.close()
    ListToCountLen = string_line.split('^')
    LenOfSQL = len(ListToCountLen)
    ListOfFor = list(range(LenOfSQL))
    string_Create_First = 'create table {tablename2}'
    string_Create_Second = ' (id TEXT primary key,'
    string_Insert_First = 'insert into {tablename2} values'
    string_Insert_Second = '(?,'
    string_AToZ = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for num1 in ListOfFor:
        string_Create_Second += string_AToZ[num1] + ' TEXT'
        string_Insert_Second += '?'
        if (num1 != ListOfFor[LenOfSQL - 1]):
            string_Create_Second += ',';
            string_Insert_Second += ','
    string_Create_Second += ')';
    string_Insert_Second += ')'
    string_Create_Full = string_Create_First + string_Create_Second
    string_Insert_Full = string_Insert_First + string_Insert_Second
    return string_Create_Full, string_Insert_Full, LenOfSQL  # 返回Create指令、Insert指令、数据行的字数据数量


def ModFun_RecvFileByTCP(string_FilePath_First, string_host, string_port, TranSpeed_Single):
    socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket1.bind((string_host, string_port))  # 绑定地址到套接字
    socket1.listen(5);
    print('等待来自客户端的连接')
    while True:
        OutSock1, addr1 = socket1.accept();
        print('收到来自', addr1, '的连接请求，即将开始传输文件')
        Recvdata1 = OutSock1.recv(1024).decode()
        RecvdataList1 = Recvdata1.split('^')
        data1 = RecvdataList1[0];
        TranSpeed_Single = RecvdataList1[1];
        string_FilePath_Second = RecvdataList1[2]  # 将传来的文件信息分离成各个变量（路径、文件名等）
        file_total_size = int(data1)
        PersentOfFileSize = file_total_size / 60;
        PersentOfFileSizeIF = PersentOfFileSize
        receivefileSize = 0;
        CntOfProgressBar = 0
        OutSock1.send('服务端开始下载文件'.encode())
        print('全部传输进度：||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n当前传输进度：', end='')
        string_file_Full = string_FilePath_First + string_FilePath_Second
        with open(string_file_Full, 'wb') as file3:  # 循环接收文件主体部分
            while receivefileSize < file_total_size:
                data2 = OutSock1.recv(int(TranSpeed_Single))
                file3.write(data2)
                receivefileSize += len(data2)
                if receivefileSize > PersentOfFileSizeIF:
                    print('|', end='')
                    PersentOfFileSizeIF += PersentOfFileSize
                    CntOfProgressBar += 1
        data_end = OutSock1.recv(1024)
        if CntOfProgressBar != 60: print('|', end='')
        if b'102455231' in data_end:
            file3.close()
            OutSock1.close()
            socket1.close()
            print('\n接收完毕，文件大小：', receivefileSize / 1000, 'KB', '；存储路径：', string_file_Full)
            return string_file_Full


def ControlCenter():
    string_file_Full1 = ModFun_RecvFileByTCP(string_StoreFilePath_First, host, port, TranSpeed_Single)
    print('正在从该文件提取数据，并存储进SQL数据库')
    ModFun_CreateAndInsertData('test3', string_file_Full1)


if __name__ == '__main__':
    Progress1 = multiprocessing.Process(target=ControlCenter, name='boyLenRecv')
    Progress1.start()
    Progress1.join()
    print('文件传输进程关闭')
