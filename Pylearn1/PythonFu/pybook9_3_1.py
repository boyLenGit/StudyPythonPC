#基本的序列和映射协议
def check_index(key):
    """
    【辅助函数】指定的键是否是可接受的索引？
    键必须是非负整数，才是可接受的。如果不是整数，
    将引发TypeError异常；如果是负数，将引发Index
    Error异常（因为这个序列的长度是无穷的）
    """
    if not isinstance(key, int):
        raise TypeError
    if key < 0:
        raise IndexError
class ArithmeticSequence:
    def __init__(self, start=0, step=1):
        """
        初始化这个算术序列
        start -序列中的第一个值
        step -两个相邻值的差
        changed -一个字典，包含用户修改后的值
        """
        self.start = start # 存储起始值
        self.step = step # 存储步长值
        self.changed = {} # 没有任何元素被修改
    def __getitem__(self, key):
        """
        从算术序列中获取一个元素
        """
        check_index(key)
        try: return self.changed[key] # 修改过？
        except KeyError: # 如果没有修改过，
            return self.start + key * self.step # 就计算元素的值
    def __setitem__(self, key, value):
        """
        修改算术序列中的元素
        """
        check_index(key)
        self.changed[key] = value # 存储修改后的值
s = ArithmeticSequence(1, 2)
print(s[4])  #调用了getitem
s[4]=1   #调用了setitem
print(s[4])
