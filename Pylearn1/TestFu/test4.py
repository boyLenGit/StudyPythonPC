class boyLen:
    class1=1
    def printWords(a):
        boyLen.class1+=1
        print(str(a)+'---'+str(boyLen.class1))
bo1=boyLen
bo1.class1=66
bo2=boyLen()
bo2.class1=77#77是bo2的属性，与boylen类的class1无关，因此boyLen的变量class1不会发生变化
bo3=boyLen()
bo3.class1=88
bo2.printWords()
bo3.printWords()
bo1.printWords('123')

