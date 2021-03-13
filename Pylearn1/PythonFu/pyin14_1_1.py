#导入txt文件，读取内容；新建txt文件，写入内容；获取某个文件的扩展名、文件名；判断字符串是否为目录；获取当前py文件的目录；获取目录下的全部文件、子文件
import os.path,os
road1='Z:/fsdf.txt'
file1 = open(road1,'w')
file1.write('1231332313')
file1.close()
list2=[]
with open('Z:\\a.txt','r+',encoding='utf-8') as file2:
    number=0
    while True:
        number+=1
        string1 = file2.readline()
        if string1=='':
            break
        list2.append(string1)
file2.close()
for strign in list2:
    print(strign,end='')
print(list2)
print(os.name)
print(os.path.splitext('Z:\\a.txt'))
print(os.path.isdir('Z:\\'))
print(os.getcwd())
#os.makedirs('Z:/er231323422df/dfsf/sdfsf')
#os.makedirs('Z:\\5434er324df\\dfsfsef.txt')
print(list(os.walk('Z:\\')))
