import requests
import json

url = 'http://httpbin.org/post'
data = {'k1': 'v1', 'k2': 'v2'}
response = requests.post(url, json.dumps(data))
print(response.text)
