# 用urllib3实现GET POST
import urllib3


def GETbaidu():
    http1 = urllib3.PoolManager()
    response1 = http1.request('GET', 'https://www.baidu.com/')
    print(response1.data.decode())


def POSTsimple():
    http1 = urllib3.PoolManager()
    response2 = http1.request('POST', 'http://httpbin.org/post', fields={'word': 'hello'})
    print(response2.data.decode())


POSTsimple()
