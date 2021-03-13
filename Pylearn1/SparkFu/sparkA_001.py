from pyspark import SparkContext
import pyspark

sc = SparkContext("local", "first app")


def learn1():
    path = 'D:\Apps\hadoop-3.3.0/README.txt'
    logdata = sc.textFile(path).cache()
    num_a = logdata.filter(lambda s: 'a' in s).count()
    num_b = logdata.filter(lambda s: 'b' in s).count()
    print(num_a, num_b)

    # 创建RDD。parallelize：将一个List列表转化为了一个RDD对象
    list1 = ["scala", "java", "hadoop", "spark", "akka", "spark vs hadoop", "pyspark", "pyspark and spark"]
    words = sc.parallelize(list1)
    list2 = [1, 2, 3, 4, 5, 6]
    numbers = sc.parallelize(list2)
    list3 = [[1, 2], [3, 4], [5, 6]]
    numbers3 = sc.parallelize(list3)

    # count()返回项目的数量
    counts = words.count()
    print('counts：', counts)

    # collect()返回所有的项
    collect1 = words.collect()
    print('collect:', collect1)

    # foreach:仅返回满足foreach内函数条件的元素。
    def f(x1):
        x1 = x1 ** 2
        print('foreach1:', x1)

    print('foreach2:', numbers.foreach(f))

    # filter：返回一个经过过滤器过滤后的RDD，过滤条件在filter的()中
    filter1 = words.filter(lambda x: 'spark' in x)
    print('filter：', filter1.collect())

    # map：通过对这个RDD的每个元素应用一个函数来返回一个新的RDD。
    map1 = words.map(lambda x: (x, 1)).collect()
    print('collect_map:', map1)

    # reduce:针对RDD对应的列表中的元素，递归地选择第一个和第二个元素进行操作，操作的结果作为一个元素用来替换这两个元素
    reduce1 = numbers.reduce(lambda a, b: a + b)
    reduce_ByKey = numbers3.reduceByKey(lambda a, b: a + b)
    reduce_ByKeyLocally = numbers3.reduceByKeyLocally(lambda a, b: a + b)
    print('reduce:', reduce1, 'reduce_ByKey:', reduce_ByKey.collect(), 'reduce_ByKeyLocally:', reduce_ByKeyLocally)

    # parallelize的第二个参数
    numbers1 = sc.parallelize(list2, 2)
    print('parallelize单分区1:', numbers.glom().collect())
    print('parallelize双分区2:', numbers1.glom().collect())

    # flatMap:功能与map一样，但返回的是一维的list，即将map的二维list拍扁
    map2 = numbers1.map(lambda x: (x, x ** 2))
    flatmap1 = numbers1.flatMap(lambda x: (x, x ** 2))
    print('map2:', map2.collect(), 'flatmap1:', flatmap1.collect())

    # join
    x = sc.parallelize([('A', 1), ('A', 2), ('B', 3), ('C', 4)])
    y = sc.parallelize([('A', 999), ('B', 888), ('B', 777), ('C', 666)])
    print('X-Y-JOIN:', x.join(y).collect())
    print('Y-X-JOIN:', y.join(x).collect())

    # union:合并并且求并集，不去掉重复元素
    x = sc.parallelize([1, 2, 3, 4])
    y = sc.parallelize([1, 22, 33, 44])
    print('union:', x.union(y).collect())
    # intersection:求两个元素的交集，即只要两者都有的元素
    print('intersection:', x.intersection(y).collect())

    # cache
    words.cache()
    print('is_cached:', words.persist().is_cached)

    # broadcast
    list2 = ["scala", "java", "hadoop", "spark"]
    bd1 = sc.broadcast(list2)
    print('bd1.value:', bd1.value, bd1.value[2])
    bd1.destroy()

    # groupBy
    x = sc.parallelize([1, 2, 3])
    y = x.groupBy(lambda x: 'A' if (x % 2 == 1) else 'B')
    y1 = y.collect()[0]
    for i in y.collect():
        for ii in i[1]:
            print('groupBy:', i, '----', ii)
    print('groupBy-y:', y.collect())

    # pipe——报错
    '''
    x1 = x.pipe('grep -i "A"')
    print('pipe:', x1.collect())'''

    # first
    print('first:', numbers.first())

    # max/min
    y = sc.parallelize([12, 222, 33, 44])
    print('max:', y.max(key=int))

    # sun
    print('sum:', y.sum())

    # count
    print('count:', y.count())

    # partitionBy
    x = sc.parallelize([(0, 1), (1, 2), (2, 3), (1, 11), (11, 22), (22, 33)], 3)
    x1 = sc.parallelize([(0, 1), (1, 2), (2, 3), (1, 11), (11, 22), (22, 33)])
    # y = x.partitionBy(numPartitions=3, partitionFunc=lambda x1: x1)
    y = x.partitionBy(4)
    y1 = x1.partitionBy(4)
    print('partitionBy_3:', x.glom().collect())
    print('partitionBy_3:', y.glom().collect())
    print('partitionBy_1:', x1.glom().collect())  # 结论：x分不分组一点也不影响partitionBy的分组
    print('partitionBy_1:', y1.glom().collect())


# combineByKey的使用
def learn2():
    # x = sc.parallelize([('B', 1), ('B', 2), ('A', 3), ('A', 4), ('A', 5)])
    x = sc.parallelize([('B', 'a'), ('B', 'b'), ('A', 'c'), ('A', 'd'), ('A', 'e'), ('C', 'c')])
    createCombiner = (lambda el: [(el, el + '-')])

    def parallelize_mergeVal(aggregated, el):
        print('---', aggregated)
        return aggregated + [(el, el + '-')]

    mergeVal = (lambda aggregated, el: aggregated + [(el, el + '-')])  # append to aggregated
    mergeComb = (lambda agg1, agg2: agg1 + agg2)  # append agg1 with agg2
    y = x.combineByKey(createCombiner, parallelize_mergeVal, mergeComb)
    print(x.collect())
    print(y.collect())


# aggregateByKey
def learn3():
    x = sc.parallelize([('B', 1), ('B', 2), ('A', 3), ('A', 4), ('A', 5)])
    zeroValue = [100]

    def mergeVal_func(aggregated, el):
        print(aggregated)
        return aggregated + [(el, el ** 2)]

    mergeVal = (lambda aggregated, el: aggregated + [(el, el ** 2)])
    mergeComb = (lambda agg1, agg2: agg1 + agg2)
    y = x.aggregateByKey(zeroValue, mergeVal_func, mergeComb)
    print(x.collect())
    print(y.collect())


# foldByKey
def learn4():
    x = sc.parallelize([('B', 1), ('B', 2), ('A', 3), ('A', 4), ('A', 5)])
    y = x.foldByKey(zeroValue=1, func=(lambda result, value: result + value))
    print(x.collect())
    print(y.collect())


# flatMapValues，分别用def与lambda进行测试比较
def learn5():
    x = sc.parallelize([('A', (1, 2, 3)), ('B', (4, 5))])

    def def_func(x1):
        x2 = []
        for i in x1:
            # print('lambda_func,i:', i)
            i = i ** 2
            x2.append(i)
        return x2

    lambda_func = lambda x1: [i ** 2 for i in x1]  # lambda和def的效果是等价的
    y_def = x.flatMapValues(def_func)
    y_lambda = x.flatMapValues(lambda_func)
    print('x:', x.collect())
    print('y_def:', y_def.collect())
    print('y_lambda:', y_lambda.collect())


# groupWith
def learn6():
    x = sc.parallelize([('C', 4), ('B', (3, 3)), ('A', 2), ('A', (1, 1))])
    y = sc.parallelize([('B', (7, 7)), ('A', 6), ('A', 9), ('D', (5, 5))])
    z = sc.parallelize([('D', 9), ('B', (8, 8))])
    a = x.groupWith(y, z)
    print('x.collect():', x.collect())
    print('y.collect():', y.collect())
    print('z.collect():', z.collect())
    print('a.collect():', a.collect())
    print('list(a.collect()):', list(a.collect()))
    print("Result:")
    for key, val in list(a.collect()):
        print('for1:', key, '---', val[0], '---', list(val[0]))
        print('for2:', key, [list(i) for i in val])


# sampleByKey
def learn7():
    x = sc.parallelize([('A', 1), ('B', 2), ('C', 3), ('B', 4), ('A', 5)])
    y = x.sampleByKey(withReplacement=False, fractions={'A': 0.5, 'B': 0.3, 'C': 0.2})
    print(x.collect())
    print(y.collect())


# subtractByKey
def learn8():
    x = sc.parallelize([('C', 1), ('B', 2), ('A', 3), ('A', 4), ('K', 4)])
    y = sc.parallelize([('A', 5), ('D', 6), ('A', 7), ('D', 8), ('E', 8)])
    z = x.subtractByKey(y)
    print('x.collect():', x.collect())
    print('y.collect():', y.collect())
    print('z.collect():', z.collect())


# keyBy:将rdd的每单项构成一个元组
def learn9():
    x = sc.parallelize([1, 2, 3])
    y = x.keyBy(lambda x1: x1 ** 2 + 100)
    print('x.collect():', x.collect())
    print('y.collect():', y.collect())


# repartition
def learn10():
    x = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 45], 2)
    y = x.repartition(numPartitions=5)
    print(x.glom().collect())
    print(y.glom().collect())


# coalesce与repartition的功能对比
def learn11():
    x = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 45], 6)
    y = x.coalesce(numPartitions=5, shuffle=True)
    y2 = x.repartition(numPartitions=5)  # repartition就是coalesce的shuffle=True的时候。
    print(x.glom().collect())
    print(y.glom().collect())
    print(y2.glom().collect())


#
def learn12():
    x = sc.parallelize(['B', 'A', 'A'])
    y = x.map(lambda x1: ord(x1))
    z = x.zip(y)
    print(x.collect())
    print(y.collect())
    print(z.collect())
    x = sc.parallelize([('B', 1), ('B', 2), ('A', 3), ('A', 4), ('A', 5)])
    print(x.count())


# sample
def learn13():
    x1 = sc.parallelize(range(10), 4)
    x2 = x1.sample(False, 0.5, 80)
    print(x1.collect())
    print(x2.collect())
    x3 = sc.parallelize(range(100))
    x4 = x3.sample(False, 0.5, 80)
    x4.setName('boyboy')
    print(x3.count(), x4.count(), x4.name())


def learn14():
    rdd1 = sc.parallelize([('a', 1), ('b', 4), ('c', 10)]).saveAsTextFile('F:/aaa/b2bb.txt')
    rdd2 = sc.parallelize([('a', 4), ('a', 1), ('b', 6), ('d', 15)])
    rdd3 = rdd1.leftOuterJoin(rdd2)
    print(rdd3.collect())


if __name__ == '__main__':
    learn1()
