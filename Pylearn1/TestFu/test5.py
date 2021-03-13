#分割文本~859~为859
string1='~859~';string2='~01001~^~305~^24.^17^0.^~1~^~A~^^^7^19.^27.^7^22.^24.^~2, 3~^~03/01/2003~'
list1=['~859~', '~g~', '~F18D1TN7~', '~18:1-11 t (18:1t n-7)~', '~3~', '12310\n']
list2=[]
list3=string2.split('^')
print('list3:',list3)
print(string1.strip('~'))
for string2 in list3:
    list2.append(string2.strip('~'))
print(list2)
list4=list2[2:]
list4.extend([list2[1]])
print(list4)
print(list2[1])