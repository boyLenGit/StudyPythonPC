# 用requests实现GET POST
import requests

url_JDWeather = 'https://way.jd.com/he/freeweather?city=北京'
dic2 = {'appkey': 'd83ecb3e3d17e71995685e263547e607'}


def get1():
    response1 = requests.get('http://www.baidu.com')
    print('状态码：', response1.status_code)
    print('请求url：', response1.url)
    print('头部信息：', response1.headers)
    print('cookie信息：', response1.cookies)
    print('文本形式的网页源码：', response1.text)
    print('字节流形式打印网页源码：', response1.content.decode())


def post1():
    data1 = {'word': 'hello'}
    response1 = requests.post('http://httpbin.org/post', data=data1)
    print(response1.content.decode())


def get_take_params():
    dic1 = {'key1': 'value1', 'key2': 'value2'}
    response1 = requests.get(url_JDWeather, params=dic2)
    print(response1.content.decode())


get_take_params()
get1()
