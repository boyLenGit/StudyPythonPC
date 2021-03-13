#SQLite数据库
import sqlite3
#创建数据库
connection1=sqlite3.connect('boylen2.db')
cursor1 = connection1.cursor()
#cursor1.execute('create table user (id int(10) primary key,name varchar(20))')
'''
cursor1.execute('insert into user (id,name) values ("1","a")')
cursor1.execute('insert into user (id,name) values ("2","b")')
cursor1.execute('insert into user (id,name) values ("3","c")')
cursor1.execute('insert into user (id,name) values ("4","d")')'''
cursor1.execute('insert into user (id,name) values ("8","h")')
#向数据库中读取数据，共用三种方法
cursor1.execute('select * from user ')
result1=cursor1.fetchone();#查一条数据
print(result1)
result2=cursor1.fetchmany(2)#查指定条数数据
print(result2)
result3=cursor1.fetchall()#查多条数据
print(result3)
#向数据库中读取数据，用占位符0（123等都可，？也可）可以避免SQL注入的风险
cursor1.execute('select * from user where id>0')
result3=cursor1.fetchall()
print('加where：',result3)

#修改数据库的信息
cursor1.execute('update user set name =?where id =?',('name为boyLn',1))
#cursor1.execute('update user set id =?where name =?',(1,'MR'))
cursor1.execute('select * from user where id>0')
result3=cursor1.fetchall()
print('改信息：',result3)
#删除数据库的信息
cursor1.execute('delete from user where id=?',(4,))
cursor1.execute('select * from user where id>0')
result3=cursor1.fetchall()
print('删信息：',result3)
#cursor1.close()
#
connection1.close()

connection1=sqlite3.connect('boylen2.db')
cursor1 = connection1.cursor()
cursor1.execute('select * from user where id>0')
result4=cursor1.fetchall()
print('关闭后再读：',result4)
cursor1.close()
connection1.close()