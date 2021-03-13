def func0(seconds):
    if seconds > 180:
        division = 0
        if (seconds - 180) % 60 != 0:
            division = 1
        money = 2 + 2 * ((seconds - 180) // 60) + 2 * division
        money = 0.2 + 0.2 * ((seconds - 180) // 60) + 0.2 * division
    else:
        money = 2
    return money / 10


def func(time):
    time = float(time)
    fee = 0
    if time / 60 <= 3:
        fee = 0.2
    elif time % 60 != 0:
        fee = 0.2 + 0.2 * ((time - 180) // 60 + 1)
    elif time % 60 == 0:
        fee = 0.2 + 0.2 * ((time - 180) // 60)
    return int(fee * 10) / 10


def func2(list1, list2):
    # 初始化
    odds = []  # 储存奇数
    evens = []  # 储存偶数
    total = []  # 储存全部数
    # 分离奇偶
    for i in range(len(list1)):
        if list1[i] % 2 != 0:  # 整除除不尽，则为奇数
            odds.append(list1[i])
        else:
            evens.append(list1[i])
    for i in range(len(list2)):
        if list2[i] % 2 != 0:  # 整除除不尽，则为奇数
            odds.append(list2[i])
        else:
            evens.append(list2[i])
    # 排序
    odds.sort()
    evens.sort()
    # 合并
    total.extend(odds)
    total.extend(evens)
    print(total)
    return total


def func3(input):
    input = str(input)
    list = []
    list2 = []
    list3 = []
    string1 = ''
    for i in range(len(input)):
        list.extend(input[i])
    list2.extend(list)
    for i in range(len(input)):
        list2[len(input) - i - 1] = list[i]
    return ' '.join(list2)


def func4():
    list_features_to_keep = ['INFANT_ALIVE_AT_REPORT', 'BIRTH_PLACE', 'MOTHER_AGE_YEARS', 'FATHER_COMBINED_AGE',
                             'CIG_1_TRI',
                             'MOTHER_HEIGHT_IN', 'MOTHER_PRE_WEIGHT', 'DIABETES_PRE', 'DIABETES_GEST', 'HYP_TENS_PRE',
                             'HYP_TENS_GEST', 'PREV_BIRTH_PRETERM']
    print([i for i in list_features_to_keep])


cnt1 = 0
cnt_little = 0


def func5():
    list1 = [[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]
    list_store_checked = list1
    global cnt1, cnt_little

    def check_heng(i, ii, list1):
        global cnt1, cnt_little
        print('check_heng:', i, ii, 'cnt1:', cnt1, len(list1), len(list1[0]), cnt_little)
        if ((i + 1) >= len(list1)) or ((ii + 1) >= len(list1[0])):
            return None  # 如果已经到行末端
        i = i + 1
        # print('Second:', i, ii, cnt1, len(list1), len(list1[0]))
        if list1[i][ii] == 0:
            if cnt_little <= 0 and list_store_checked[i][ii] != -1:
                cnt1 = cnt1 + 1
                print('-----cnt check1:', i, ii)
                return None
            cnt_little = cnt_little - 1
            return None

        if list1[i][ii] == 1 and list_store_checked[i][ii] != -1:
            list_store_checked[i][ii] = -1  # 标记
            list1[i][ii] = cnt1  # 记录
            if cnt_little >= 0: check_heng(i, ii, list1);cnt_little = cnt_little + 1
            if cnt_little >= 0: check_shu(i, ii, list1);cnt_little = cnt_little + 1
            if cnt_little >= 0: check_shuFan(i, ii, list1);cnt_little = cnt_little + 1

    def check_shu(i, ii, list1):
        global cnt1, cnt_little
        print('check_heng:', i, ii, 'cnt1:', cnt1, len(list1), len(list1[0]), cnt_little)
        if ((i + 1) >= len(list1)) or ((ii + 1) >= len(list1[0])):
            return None  # 如果已经到行末端
        ii = ii + 1
        if list1[i][ii] == 0:
            if cnt_little <= 0 and list_store_checked[i][ii] != -1:
                print('-----cnt check2:', i, ii)
                cnt1 = cnt1 + 1
                return None
            cnt_little = cnt_little - 1
            return None

        if list1[i][ii] == 1 and list_store_checked[i][ii] != -1:
            list_store_checked[i][ii] = -1  # 标记
            list1[i][ii] = cnt1  # 记录
            if cnt_little >= 0: check_heng(i, ii, list1);cnt_little = cnt_little + 1
            if cnt_little >= 0: check_shu(i, ii, list1);cnt_little = cnt_little + 1
            if cnt_little >= 0: check_shuFan(i, ii, list1);cnt_little = cnt_little + 1

    def check_shuFan(i, ii, list1):
        global cnt1, cnt_little
        print('check_heng:', i, ii, 'cnt1:', cnt1, len(list1), len(list1[0]), cnt_little)
        if ((i + 1) >= len(list1)) or ((ii + 1) >= len(list1[0])):
            return None  # 如果已经到行末端
        if ii == 0: return None
        ii = ii - 1
        if list1[i][ii] == 0:
            if cnt_little <= 0 and list_store_checked[i][ii] != -1:
                print('-----cnt check3:', i, ii)
                cnt1 = cnt1 + 1
                return None
            cnt_little = cnt_little - 1
            return None

        if list1[i][ii] == 1 and list_store_checked[i][ii] != -1:
            list_store_checked[i][ii] = -1  # 标记
            list1[i][ii] = cnt1  # 记录
            if cnt_little >= 0: check_heng(i, ii, list1);cnt_little = cnt_little + 1
            if cnt_little >= 0: check_shu(i, ii, list1);cnt_little = cnt_little + 1
            if cnt_little >= 0: check_shuFan(i, ii, list1);cnt_little = cnt_little + 1

    for i in range(len(list1)):
        for ii in range(len(list1[0])):
            cnt_little = 0
            if list1[i][ii] == 1 and list_store_checked[i][ii] != -1:  # 如果是1且没算过
                list_store_checked[i][ii] = -1  # 标记
                list1[i][ii] = cnt1
                if (i - 1) != len(list1): check_heng(i, ii, list1)
                if (i - 1) != len(list1[0]): check_shu(i, ii, list1)
    print(cnt1)
    return cnt1


def func6(input1):
    dict1 = {}
    list_name = []
    for i1 in input1:
        if i1 not in list_name:
            list_name.append(i1)
    for i2 in list_name:
        dict1[i2] = 0
    for i3 in input1:
        dict1[i3] = dict1[i3] + 1
    cnt_max = 0
    for i4 in dict1.keys():
        if dict1[i4] > cnt_max:
            cnt_max = dict1[i4]
    dict2 = {}
    for i5 in dict1.keys():
        if dict1[i5] == cnt_max:
            dict2[i5] = cnt_max
    print('重复最多元素：{0}'.format(str(list(dict2.keys()))) + ',重复个数：{0}'.format(str(list(dict2.values()))))
    return ('重复最多元素：{0}'.format(str(list(dict2.keys()))) + ',重复个数：{0}'.format(str(list(dict2.values()))))


def func7():
    cnt = 0
    for i in range(3, 10):
        cnt += 1
        print(i, cnt, '|', end=' ')


def func8():
    list1 = [['max', [2475.0, 3205.0, 3138.0, 3356.0, 2613.0, 3543.0]], ['count', [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]],
             ['75%', [906.0, 1381.0, 1480.0, 498.0, 677.0, 1706.0]]]
    for i1 in range(len(list1)):
        list1[i1][0] = list1[i1][0].replace('%', 'Per').replace('count', 'Count').replace('max', 'Max').replace('min',
                                                                                                                'Min')
    print(list1)
    print('----2----')
    list1 = [['max', [2475.0, 3205.0, 3138.0, 3356.0, 2613.0, 3543.0]], ['count', [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]],
             ['75%', [906.0, 1381.0, 1480.0, 498.0, 677.0, 1706.0]]]
    list1_1 = []
    for i1 in range(len(list1)):
        list1_1.extend(list1[i1][1])
    list1_1.sort()
    print(list1_1[0], list1_1[-1])


if __name__ == '__main__':
    list1 = [1, 2, 3]
    print(list1[0:3])
