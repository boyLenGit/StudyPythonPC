import json

import requests

url_wlan = "http://121.251.251.207/eportal/InterFace.do?method=login"
url_upc_guard = 'http://stu.gac.upc.edu.cn:8089/stuqj/addQjMess'

# ----------------------------------------------------- header -------------------------------------------------------
header_wlan = {'Host': '121.251.251.207',
               'Connection': 'keep-alive',
               'Content-Length': '584',
               'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36 Edg/86.0.622.61',
               'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
               'Accept': '*/*',
               'Origin': 'http://121.251.251.207',
               'Referer': 'http://121.251.251.207/eportal/index.jsp?wlanuserip=180.201.132.2&wlanacname=&nasip=172.22.242.21&wlanparameter=c8-d3-ff-df-41-43&url=http://www.upc.edu.cn/&userlocation=ethtrunk/62:1862.0',
               'Accept-Encoding': 'gzip, deflate',
               'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,zh-TW;q=0.5',
               'Cookie': 'EPORTAL_COOKIE_OPERATORPWD=; EPORTAL_AUTO_LAND=; EPORTAL_COOKIE_DOMAIN=true; EPORTAL_COOKIE_SAVEPASSWORD=true; EPORTAL_COOKIE_PASSWORD=7f615b7c4733198836559e02593d4d765d7c0da86bb516fe809401f6ebcbf68a2578ee1b0746eeada44215f5d8e97f3abfd00a93fd6a16061ce9b1396fccb506bb6dcbda7294019aa711c12a5881279b067d50ab1e91429fef0c54de31939b1936c11b93eea7e12d44c606c54a65b44e45422e03caa178c97e7c89dc5e915808; EPORTAL_COOKIE_USERNAME=Z20050020; EPORTAL_COOKIE_SERVER=cmcc; EPORTAL_COOKIE_SERVER_NAME=%E4%B8%AD%E5%9B%BD%E7%A7%BB%E5%8A%A8; EPORTAL_USER_GROUP=%E7%A0%94%E7%A9%B6%E7%94%9F%E7%BB%84; JSESSIONID=39700723542F988603A7A22F5E29D1BA'
               }

header_upc_guard = {'Host': 'stu.gac.upc.edu.cn:8089', 'Accept': '*/*', 'X-Requested-With': 'XMLHttpRequest',
                    'Accept-Language': 'zh-cn', 'Accept-Encoding': 'gzip, deflate',
                    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                    'Origin': 'http://stu.gac.upc.edu.cn:8089',
                    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 MicroMessenger/7.0.18(0x17001233) NetType/4G Language/zh_CN',
                    'Connection': 'keep-alive',
                    'Referer': 'http://stu.gac.upc.edu.cn:8089/xswc?code=67eb0f06c90b79f32b8cc055f6ccc479&state=2',
                    'Content-Length': '483'}

# ----------------------------------------------------- json -------------------------------------------------------
json_wlan = {"userId": "Z20050020",
             "password": "7f615b7c4733198836559e02593d4d765d7c0da86bb516fe809401f6ebcbf68a2578ee1b0746eeada44215f5d8e97f3abfd00a93fd6a16061ce9b1396fccb506bb6dcbda7294019aa711c12a5881279b067d50ab1e91429fef0c54de31939b1936c11b93eea7e12d44c606c54a65b44e45422e03caa178c97e7c89dc5e915808",
             "service": "cmcc",
             "queryString": "wlanuserip%253D180.201.132.2%2526wlanacname%253D%2526nasip%253D172.22.242.21%2526wlanparameter%253Dc8-d3-ff-df-41-43%2526url%253Dhttp%253A%252F%252Fwww.upc.edu.cn%252F%2526userlocation%253Dethtrunk%252F62%253A1862.0",
             "operatorPwd": "",
             "operatorUserId": "",
             "validcode": "",
             "passwordEncrypt": "true"
             }
json_upc_guard = {'stuXh': 'Z20050020',
                  'stuXm': '马博仑',
                  'stuXy': '控制科学与工程学院',
                  'stuZy': '电子信息',
                  'stuMz': '汉族',
                  'stuBj': '研控2003',
                  'stuLxfs': '13012425528',
                  'stuJzdh': '18265615965',
                  'stuJtfs': '步行',
                  'stuStartTime': '2021-01-20+10:28:00',
                  'stuReason': '办理业务',
                  'stuWcdz': '黄岛区',
                  'stuJjlxr': '马显智',
                  'stuJjlxrLxfs': '18265615965'}
json_upc_guard2 = {'stuXh': 'Z20050020',
                   'stuXm': '%E9%A9%AC%E5%8D%9A%E4%BB%91',
                   'stuXy': '%E6%8E%A7%E5%88%B6%E7%A7%91%E5%AD%A6%E4%B8%8E%E5%B7%A5%E7%A8%8B%E5%AD%A6%E9%99%A2',
                   'stuZy': '%E7%94%B5%E5%AD%90%E4%BF%A1%E6%81%AF',
                   'stuMz': '%E6%B1%89%E6%97%8F',
                   'stuBj': '%E7%A0%94%E6%8E%A72003',
                   'stuLxfs': '13012425528',
                   'stuJzdh': '18265615965',
                   'stuJtfs': '%E6%AD%A5%E8%A1%8C',
                   'stuStartTime': '2021-01-18+10%3A28%3A00&',
                   'stuReason': '%E5%8A%9E%E7%90%86%E4%B8%9A%E5%8A%A1',
                   'stuWcdz': '%E9%BB%84%E5%B2%9B%E5%8C%BA',
                   'stuJjlxr': '%E9%A9%AC%E6%98%BE%E6%99%BA',
                   'stuJjlxrLxfs': '18265615965'}


response_upc_guard = requests.post(url=url_upc_guard, data=json_upc_guard, headers=header_upc_guard)
# response_upc_guard = requests.post(url=url_wlan, json=json_wlan, headers=header_wlan)
'''session1 = requests.Session()
response_upc_guard = session1.post(url=url_wlan, data=json.dumps(json_wlan), headers=header_wlan)
print(response_upc_guard.text, response_upc_guard.headers)
print(response_upc_guard.content)
print(response_upc_guard.apparent_encoding)'''
print(response_upc_guard.text)

