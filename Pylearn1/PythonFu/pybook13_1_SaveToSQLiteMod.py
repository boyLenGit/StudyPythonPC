import sqlite3
def TextCuter1(stringtextcuter):
    return stringtextcuter.split('^')

def TextCuter2(list2):
    listTextCuter1 = []              #如果是全局变量的话，listTextCuter1会一直增大，这里用作每次循环清零1次
    for stringtextCuter1 in list2:
        listTextCuter1.append(stringtextCuter1.strip('~').strip('\n'))
    return listTextCuter1

def CreateAndInsertData_6NUM(WhatFileName,WhereOpen,WhatExecuteCreate,WhatExecuteInsert,tablename):
    connection1 = sqlite3.connect(WhatFileName)
    cursor1 = connection1.cursor()
    print(WhatExecuteCreate)
    cursor1.execute(WhatExecuteCreate.format(tablename=tablename))
    query = WhatExecuteInsert.format(tablename=tablename)         # 导入数据库用
    for line in open(WhereOpen):
        fields = TextCuter1(line)
        vals = TextCuter2(fields)
        cursor1.execute(query, vals)
    connection1.commit()
    connection1.close()

#更新
def GetTxtAndUpdateSQLite_6NUM(WhatFileName,WhereOpen,WhatExecute,tablename):
    connection1 = sqlite3.connect(WhatFileName)
    cursor1 = connection1.cursor()
    query2 = WhatExecute.format(tablename=tablename)
    for line in open(WhereOpen):
        fields =TextCuter1(line)
        vals = TextCuter2(fields);
        list1=[vals[1],vals[2],vals[3],vals[4],vals[5],vals[0]]
        cursor1.execute(query2, list1)
    print('SQLite更新完毕,最后一行为：',list1)
    connection1.commit()
    connection1.close()

def CreateAndInsertData_64W(tablename64W):
    id1=1
    connection1 = sqlite3.connect('food.db')
    cursor1 = connection1.cursor()
    try:
        cursor1.execute('create table {tablename2} (id TEXT primary key,A TEXT,B TEXT,C TEXT,D TEXT,E TEXT,F TEXT,G TEXT,'
                        'H TEXT,I TEXT,J TEXT,K TEXT,L TEXT,M TEXT,N TEXT,O TEXT,P TEXT,Q TEXT)'.format(tablename2=tablename64W))
    except sqlite3.OperationalError:
        print('数据表 {tablename2} 已经存在！'.format(tablename2=tablename64W))
    try:
        query = 'insert into {tablename2} values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'.format(tablename2=tablename64W)  # 导入数据库用
        for line in open('D:\\boyLen\\py\\Pylearn1\\dataLen\\NUT_DATA_BIG.txt'):
            fields = line.split('^')
            vals = TextCuter2(fields);  # print('vals:',vals)
            list_21W = [str(id1)];
            id1 += 1;
            list_21W.extend(vals)
            cursor1.execute(query, list_21W)
        print(str(id1), '条数据写入完成，最后一条为：', str(list_21W))
    except sqlite3.IntegrityError:
        print('数据表 {tablename2} 已经写入数据！错误码：UNIQUE constraint failed: NUT2.id'.format(tablename2=tablename64W))
    connection1.commit()
    connection1.close()

if __name__ =='__main__':
    print(TextCuter2(TextCuter1('~01001~^~208~^717.^0^^~4~^~NC~^^^^^^^^^^~08/01/2010~')))
    #GetTxtAndUpdateSQLite_6NUM('food.db','D:\\boyLen\\py\\Pylearn1\\dataLen\\NUTR_DEF2.txt','update {tablename} set A=?,B=?,C=?,D=?,E=? where id =?','food4')
    #CreateAndInsertData_6NUM('food.db','D:\\boyLen\\py\\Pylearn1\\dataLen\\NUTR_DEF2.txt','create table {tablename} (id TEXT primary key,A TEXT,B TEXT,C TEXT,D TEXT,E TEXT)','insert into {tablename} values(?,?,?,?,?,?)','food')