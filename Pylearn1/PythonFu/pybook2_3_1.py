#列表的生成与消除  P29
import math
print(list('hello'))
print(''.join(['h', 'e', 'l', 'l', 'o']))
print("{foo} {} {bar}  {bar} {}".format(1, 2, bar=4, foo=3))
print("{{:#g}}".format(42111))
string1 = 'Abc abc Cba'
print(string1.find('Cba'))
print(string1.title())
print(ascii('%%c'))
#字典的用法
phonebook = {'Alice': '2341', 'Beth': '9102', 'Cecil': '3258'}  #只能Cecil搜3258，不能3258搜Cecil
print(phonebook['Cecil'])
print('Abcdefg{}'.format(1))
list1=[('url', 'http://www.python.org'), ('spam', 0), ('title', 'Python Web Site')]
print(list1[0][1])