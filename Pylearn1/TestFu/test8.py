#测试一种自动构建SQLite数据库指令的方法
string_file = 'D:\\boyLen\\py\\Pylearn1\\dataLen\\NUTR_DEF2.txt'

def SmartBuildSQLiteExecute():
    with open(string_file,'r') as file1:
        string_line = file1.readline()
        file1.close()
    ListToCountLen = string_line.split('^')
    LenOfSQL = len(ListToCountLen)
    ListOfFor = list(range(LenOfSQL))
    print('ListToCountLen:',ListToCountLen,'ListOfFor:',ListOfFor,'LenOfSQL:',LenOfSQL)
    string_Create_First = 'create table {tablename2}'
    string_Create_Second =  ' (id TEXT primary key,'
    string_Insert_First = 'insert into {tablename2} values'
    string_Insert_Second = '(?,'
    string_AToZ='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for num1 in ListOfFor :
        string_Create_Second += string_AToZ[num1] + ' TEXT'
        string_Insert_Second += '?'
        if(num1 != ListOfFor[LenOfSQL-1]):
            string_Create_Second += ','
            string_Insert_Second += ','
    string_Create_Second += ')'
    string_Insert_Second += ')'
    string_Create_Full = string_Create_First + string_Create_Second
    string_Insert_Full = string_Insert_First + string_Insert_Second
    print(string_Create_Full)
    print(string_Insert_Full)
    return string_Create_Full,string_Insert_Full

def ModFun_SmartBuildSQLiteExecute(string_file):
    with open(string_file,'r') as file1:
        string_line = file1.readline()
        file1.close()
    ListToCountLen = string_line.split('^')
    LenOfSQL = len(ListToCountLen)
    ListOfFor = list(range(LenOfSQL))
    print('ListToCountLen:',ListToCountLen,'ListOfFor:',ListOfFor,'LenOfSQL:',LenOfSQL)
    string_Create_First = 'create table {tablename2}'
    string_Create_Second =  ' (id TEXT primary key,'
    string_Insert_First = 'insert into {tablename2} values'
    string_Insert_Second = '(?,'
    string_AToZ='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for num1 in ListOfFor :
        string_Create_Second += string_AToZ[num1] + ' TEXT'
        string_Insert_Second += '?'
        if(num1 != ListOfFor[LenOfSQL-1]):
            string_Create_Second += ','
            string_Insert_Second += ','
    string_Create_Second += ')'
    string_Insert_Second += ')'
    string_Create_Full = string_Create_First + string_Create_Second
    string_Insert_Full = string_Insert_First + string_Insert_Second
    print(string_Create_Full)
    print(string_Insert_Full)
    return string_Create_Full,string_Insert_Full

if __name__ == '__main__':
    SmartBuildSQLiteExecute()
    a='create table {tablename2} (id TEXT primary key,A TEXT,B TEXT,C TEXT,D TEXT,E TEXT,F TEXT,G TEXT,H TEXT,I TEXT,J TEXT,K TEXT,L TEXT,M TEXT,N TEXT,O TEXT,P TEXT,Q TEXT)'
    b = 'insert into {tablename2} values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'
    print(a)
    print(b)
