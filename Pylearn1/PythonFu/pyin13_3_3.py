import os;import time
bo=os.stat('Z:/a.txt')
time1 = time
print(bo.st_size,bo.st_dev,time1.asctime())
import pymysql
conn = pymysql.connect()