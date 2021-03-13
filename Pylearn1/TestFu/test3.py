def multiplier(factor):
    def multiplyByFactor(number):
        return number * factor
    print('HERE')
    return multiplyByFactor

a=multiplier(3)
print(a(6));print(a(7))
print(list(map(str,range(0,10))))
print(ascii(100))
def func(x):
    return 'foo' in x
seq = ["foo", "x41", "?!", "***"]
print(list(filter(func, seq)))
print(filter(func, seq))
list4=[1,2,3.3]
list44=(1,2,3.3)
print(sum(list44))
def PrintWordForModule(a):
    print(str(a)*4+'模块引用成功')
PrintWordForModule('&')