# 用urllib获取网页内容HTTP与保存本地txt
import urllib.request;
import urllib.parse
from multiprocessing import Process

string_http = 'http://news.baidu.com/'
string_file = 'Z:/test.txt'


def ReadHttpContent():
    response1 = urllib.request.urlopen(string_http)  # GET请求
    print('response:', response1)
    html1 = response1.read()
    html_decode1 = html1.decode()
    return html_decode1


def SaveHttpToTxt(html_decode):
    with open(string_file, 'w') as file1:
        file1.write(html_decode)
        file1.close()
    print('Save to txt done')


def fuse1():
    string_store_ache = ReadHttpContent()
    SaveHttpToTxt(string_store_ache)


def POSTtoHTTP():  # POST请求
    data = bytes(urllib.parse.urlencode({'word': 'hello'}), encoding='utf8')
    response1 = urllib.request.urlopen('http://httpbin.org/post', data=data)
    html1 = response1.read()
    print(html1.decode())


if __name__ == '__main__':
    process1 = Process(target=fuse1, name='boyLenHTTP')
    process1.start()
    POSTtoHTTP()
