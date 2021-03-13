#将美国农业部的数据导入到SQLite中，过程： 建立——查询——修改——删除
import sqlite3;import sys
field_count = 10;
tablename='food4';tablename21W='NUT2'

#裁剪文本函数：  用于去掉~ ：value=~859~ , value.strip('~') =859
def TextCuter1(value):
    if value.startswith('~'):
        return value.strip('~')       #用于去掉~ ：value=~859~ , value.strip('~') =859
    if not value:
        value = '0'
    return float(value)

#裁剪文本函数：将['~859~', '~g~', '~F18D1TN7~', '~18:1-11 t (18:1t n-7)~', '~3~', '12310\n'] 裁剪为['858', 'g', 'F22D4', '22:4', '3', 15160.0]
def TextCuter2(list2):
    listTextCuter1 = []   #如果是全局变量的话，listTextCuter1会一直增大，这里用作每次循环清零1次
    for stringtextCuter1 in list2:
        listTextCuter1.append(stringtextCuter1.strip('~').strip('\n'))
    return listTextCuter1

def CreateAndInsertData():
    connection1 = sqlite3.connect('food.db')
    cursor1 = connection1.cursor()
    cursor1.execute('create table {tablename} (id TEXT primary key,A TEXT,B TEXT,C TEXT,D TEXT,E TEXT)'.format(tablename=tablename))
    query = 'insert into {tablename} values(?,?,?,?,?,?)'.format(tablename=tablename)         # 导入数据库用
    for line in open('D:\\boyLen\\py\\Pylearn1\\dataLen\\NUTR_DEF2.txt'):
        fields = line.split('^')
        vals = [TextCuter1(f) for f in fields[:field_count]]
        print(vals)
        cursor1.execute(query, vals)
    connection1.commit()
    connection1.close()
# --------------------------------------------------------------------------------------------------------------------------------------------------
def CreateAndInsertData_64W():
    id1=1
    connection1 = sqlite3.connect('food.db')
    cursor1 = connection1.cursor()
    try:
        cursor1.execute('create table {tablename2} (id TEXT primary key,A TEXT,B TEXT,C TEXT,D TEXT,E TEXT,F TEXT,G TEXT,'
                        'H TEXT,I TEXT,J TEXT,K TEXT,L TEXT,M TEXT,N TEXT,O TEXT,P TEXT,Q TEXT)'.format(tablename2=tablename21W))  #1个ID 17个字母
    except sqlite3.OperationalError:
        print('数据表 {tablename2} 已经存在！'.format(tablename2=tablename21W))
    try:
        query = 'insert into {tablename2} values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'.format(tablename2=tablename21W)  # 导入数据库用 18个问号
        for line in open('D:\\boyLen\\py\\Pylearn1\\dataLen\\NUT_DATA_BIG.txt'):
            fields = line.split('^')
            vals = TextCuter2(fields);  # print('vals:',vals)
            list_21W = [str(id1)];
            id1 += 1;
            list_21W.extend(vals)
            cursor1.execute(query, list_21W)
        print(str(id1), '条数据写入完成，最后一条为：', str(list_21W))
    except sqlite3.IntegrityError:
        print('数据表 {tablename2} 已经写入数据！错误码：UNIQUE constraint failed: NUT2.id'.format(tablename2=tablename21W))
    connection1.commit()
    connection1.close()

def InsertData():
    connection1 = sqlite3.connect('food.db')
    cursor1 = connection1.cursor()
    query = 'insert into {tablename} values(?,?,?,?,?,?)'.format(tablename=tablename)   # 导入数据库用
    try:
        cursor1.execute('create table {tablename} (id TEXT primary key,A TEXT,B TEXT,C TEXT,D TEXT,E TEXT)'.format(tablename=tablename))
    except sqlite3.OperationalError:
        print('table {} 已经存在！'.format(tablename))
    for line in open('D:\\boyLen\\py\\Pylearn1\\dataLen\\NUTR_DEF2.txt'):
        fields = line.split('^')
        vals = TextCuter2(fields);
        try:
            cursor1.execute(query, vals)
            connection1.commit()
        except sqlite3.IntegrityError:
            print('^',vals[0],end='')
    print(str(vals[0]), '条数据写入完成，最后一条数据为：', vals)
    connection1.commit()
    connection1.close()

def UpdateData():  #已存在的才可以更新，不存在的不可以更新
    connection1 = sqlite3.connect('food.db')
    cursor1 = connection1.cursor()
    query2 = 'update {tablename} set A=?,B=?,C=?,D=?,E=? where id =?'.format(tablename=tablename)    #更新数据库用
    for line in open('D:\\boyLen\\py\\Pylearn1\\dataLen\\NUTR_DEF2.txt'):  #line：~852~^~g~^~F20D3N3~^~20:3 n-3~^~3~^14500
        fields = line.split('^')                                           #fields：['~859~', '~g~', '~F18D1TN7~', '~18:1-11 t (18:1t n-7)~', '~3~', '12310\n']
        vals = TextCuter2(fields);                                         #vals：['858', 'g', 'F22D4', '22:4', '3', 15160.0]
        list1=vals[1:];  list1.extend([vals[1]])                           #将ID放在最后面，与excute指令的where id位置相匹配   list1=[vals[1],vals[2],vals[3],vals[4],vals[5],vals[0]]
        cursor1.execute(query2, list1)
    print(vals[0],'条数据更新完成，最后一条数据为：', list1)
    connection1.commit()
    connection1.close()

def FindDataByID(id1):
    connection1 = sqlite3.connect('food.db')
    cursor1 = connection1.cursor()
    query2 = 'select * from {tablename} where id={id}'.format(tablename=tablename,id=str(id1))  # 更新数据库用
    cursor1.execute(query2)
    list1=cursor1.fetchall()
    print('查询ID的结果是：',list1)
    connection1.commit()
    connection1.close()

def FindDataByOther(name1,content1):
    connection1 = sqlite3.connect('food.db')
    cursor1 = connection1.cursor()
    query2 = 'select * from {tablename} where {name1}={content1}'.format(tablename=tablename,name1=str(name1),content1=str(content1))   # 更新数据库用
    print('查询内容的结果是：',query2)
    cursor1.execute(query2)
    list1=cursor1.fetchall()
    print(list1)
    connection1.commit()
    connection1.close()

def DeleteDataByID(id1):   #根据ID删除一整行数据
    connection1 = sqlite3.connect('food.db')
    cursor1 = connection1.cursor()
    query2 = 'delete  from {tablename} where id={id}'.format(tablename=tablename,id=str(id1))  # 更新数据库用
    cursor1.execute(query2)
    connection1.commit()
    connection1.close()
    print('删除的内容ID是：', id1)

def InQuireMessageOfData():
    connection1 = sqlite3.connect('food.db')
    cursor1 = connection1.cursor()
    print(cursor1.execute('tables'))
    connection1.commit()
    connection1.close()

if __name__ =='__main__':
    #CreateAndInsertData()
    InsertData()
    UpdateData()
    FindDataByID(209)
    FindDataByOther('E','600')
    FindDataByOther('B',"'FAT'")
    #InQuireMessageOfData()
    CreateAndInsertData_64W()
    DeleteDataByID(205)