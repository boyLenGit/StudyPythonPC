class Big:
    b=1
    def BigFun(self,a):
        a+=1;print(Big.b)
    def BigFun2(self,a):
         a+=1;print(Big.b);
class Lite(Big):
    def LiteFun(self,a):
        a+=1;
        b=a;print(b)
te=Lite()
te.LiteFun(1)
te.BigFun2(1)
te.BigFun(1)
#self的变量似乎是全局更改的
class Calculator:
    def calculate(self, expression):
        self.value = eval(expression)
class Talker:
    def talk(self):
        print('Hi, my value is', self.value)
class TalkingCalculator(Calculator, Talker):
    pass
tc = TalkingCalculator()
tc.calculate('1 + 2 * 3')
tc.talk()