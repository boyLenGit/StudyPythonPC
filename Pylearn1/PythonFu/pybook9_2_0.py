class FooBar:
    def __init__(self, value=42):
        self.somevar = value
        self.abc=1
class boylen(FooBar):
    def testFun(self):
        print('1'+str(self.abc))
f = FooBar('This is a constructor argument')
print(f.somevar)
bo=boylen(1)
print('1'+      str(bo.testFun()))
import tensorflow
print(tensorflow.__path__)
import sys
print(sys.path)
from TestFu import test3

test3.PrintWordForModule('12334345')
from PythonFu import pyin13_6_1

pyin13_6_1.CountS_B()