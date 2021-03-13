import segyio


# SegyFile验证
def learn5():
    path2 = 'E:/Research/data/F3_entire.segy'
    with segyio.open(path2, mode='r+') as F3_entire:
        segy_text_all = F3_entire.text[0]
        print('######text:', F3_entire.text[0])
    F3_entire.close()
    print(type(str(segy_text_all)))


if __name__ == '__main__':
    learn5()