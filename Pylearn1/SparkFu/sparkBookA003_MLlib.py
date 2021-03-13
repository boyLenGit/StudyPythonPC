# GraphFrame功能验证
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import functions
from pyspark.sql import SQLContext
import pyspark
import databricks
import pyspark.sql as sql
import matplotlib.pyplot as plt
from bokeh import charts
from bokeh.io import output_notebook
import graphframes
import pyspark.sql.types as typ
import pyspark.mllib.stat as stat
import numpy

sc = SparkContext("local", "first app")
ss = SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option",
                                                                       "some-value").getOrCreate()


# 加载和转换数据模块
def learn1():
    # 定义schema1与每列的数据类型
    labels = [
        ('INFANT_ALIVE_AT_REPORT', typ.StringType()), ('BIRTH_YEAR', typ.IntegerType()),
        ('BIRTH_MONTH', typ.IntegerType()), ('BIRTH_PLACE', typ.StringType()),
        ('MOTHER_AGE_YEARS', typ.IntegerType()), ('MOTHER_RACE_6CODE', typ.StringType()),
        ('MOTHER_EDUCATION', typ.StringType()), ('FATHER_COMBINED_AGE', typ.IntegerType()),
        ('FATHER_EDUCATION', typ.StringType()), ('MONTH_PRECARE_RECODE', typ.StringType()),
        ('CIG_BEFORE', typ.IntegerType()), ('CIG_1_TRI', typ.IntegerType()),
        ('CIG_2_TRI', typ.IntegerType()), ('CIG_3_TRI', typ.IntegerType()),
        ('MOTHER_HEIGHT_IN', typ.IntegerType()), ('MOTHER_BMI_RECODE', typ.IntegerType()),
        ('MOTHER_PRE_WEIGHT', typ.IntegerType()), ('MOTHER_DELIVERY_WEIGHT', typ.IntegerType()),
        ('MOTHER_WEIGHT_GAIN', typ.IntegerType()), ('DIABETES_PRE', typ.StringType()),
        ('DIABETES_GEST', typ.StringType()), ('HYP_TENS_PRE', typ.StringType()),
        ('HYP_TENS_GEST', typ.StringType()), ('PREV_BIRTH_PRETERM', typ.StringType()),
        ('NO_RISK', typ.StringType()), ('NO_INFECTIONS_REPORTED', typ.StringType()),
        ('LABOR_IND', typ.StringType()), ('LABOR_AUGM', typ.StringType()),
        ('STEROIDS', typ.StringType()), ('ANTIBIOTICS', typ.StringType()),
        ('ANESTHESIA', typ.StringType()), ('DELIV_METHOD_RECODE_COMB', typ.StringType()),
        ('ATTENDANT_BIRTH', typ.StringType()), ('APGAR_5', typ.IntegerType()),
        ('APGAR_5_RECODE', typ.StringType()), ('APGAR_10', typ.IntegerType()),
        ('APGAR_10_RECODE', typ.StringType()), ('INFANT_SEX', typ.StringType()),
        ('OBSTETRIC_GESTATION_WEEKS', typ.IntegerType()), ('INFANT_WEIGHT_GRAMS', typ.IntegerType()),
        ('INFANT_ASSIST_VENTI', typ.StringType()), ('INFANT_ASSIST_VENTI_6HRS', typ.StringType()),
        ('INFANT_NICU_ADMISSION', typ.StringType()), ('INFANT_SURFACANT', typ.StringType()),
        ('INFANT_ANTIBIOTICS', typ.StringType()), ('INFANT_SEIZURES', typ.StringType()),
        ('INFANT_NO_ABNORMALITIES', typ.StringType()), ('INFANT_ANCEPHALY', typ.StringType()),
        ('INFANT_MENINGOMYELOCELE', typ.StringType()), ('INFANT_LIMB_REDUCTION', typ.StringType()),
        ('INFANT_DOWN_SYNDROME', typ.StringType()), ('INFANT_SUSPECTED_CHROMOSOMAL_DISORDER', typ.StringType()),
        ('INFANT_NO_CONGENITAL_ANOMALIES_CHECKED', typ.StringType()), ('INFANT_BREASTFED', typ.StringType())
    ]

    # 设定读取文件的路径，并将数据转成DF
    path1 = 'E:/Research/data/births_train.csv.gz'
    schema1 = typ.StructType([typ.StructField(e[0], e[1], False) for e in labels])
    df_births = ss.read.csv(path=path1, header=True, schema=schema1)

    # 将selected_features从df中提取出来，并形成新的df
    selected_features = ['INFANT_ALIVE_AT_REPORT', 'BIRTH_PLACE', 'MOTHER_AGE_YEARS', 'FATHER_COMBINED_AGE',
                         'CIG_BEFORE', 'CIG_1_TRI', 'CIG_2_TRI', 'CIG_3_TRI', 'MOTHER_HEIGHT_IN', 'MOTHER_PRE_WEIGHT',
                         'MOTHER_DELIVERY_WEIGHT', 'MOTHER_WEIGHT_GAIN', 'DIABETES_PRE', 'DIABETES_GEST',
                         'HYP_TENS_PRE', 'HYP_TENS_GEST', 'PREV_BIRTH_PRETERM']
    births_trimmed = df_births.select(selected_features)

    # 调用写的correct_fig1函数对df指定列的内容进行修改：当feat！=99时，返回原值；否则返回0.
    def correct_fig1(feat):
        return functions.when(functions.col(feat) != 99, functions.col(feat)).otherwise(0)

    births_transformed = births_trimmed \
        .withColumn('CIG_BEFORE', correct_fig1('CIG_BEFORE')) \
        .withColumn('CIG_1_TRI', correct_fig1('CIG_1_TRI')) \
        .withColumn('CIG_2_TRI', correct_fig1('CIG_2_TRI')) \
        .withColumn('CIG_3_TRI', correct_fig1('CIG_3_TRI'))
    births_transformed.show(3)

    # 提取出births_trimmed中的全部schema列名与数据类型，构成一个元组，例如：('INFANT_ALIVE_AT_REPORT', StringType)
    cols = [(col.name, col.dataType) for col in births_trimmed.schema]

    # 将所有包含‘Y’的列的列名提取出来，存到一个list中
    YNU_cols = []
    for i, s in enumerate(cols):
        if s[1] == typ.StringType():
            dis = df_births.select(s[0]).rdd.map(lambda row: row[0]).collect()
            if 'Y' in dis:
                YNU_cols.append(s[0])

    # 编码替换内容用的字典：
    recode_dictionary = {'YNU': {'Y': 1, 'N': 0, 'U': 0}}

    # recode1函数从字典recode_dictionary中返回特定的键-值
    def recode1(col, key):
        return recode_dictionary[key][col]

    # udf是用户自定义函数。该函数直接返回一个Column，用于在后面构成df1的第二列。
    rec_integer = functions.udf(recode1, typ.IntegerType())

    # 将data_births中含有N Y的列全部替换成1 0。rec_integer部分原理未知。
    # df1 = df_births.select('INFANT_NICU_ADMISSION', rec_integer('INFANT_NICU_ADMISSION', functions.lit('YNU')).alias('INFANT_NICU_ADMISSION_RECODE'))
    df1 = df_births.select('INFANT_NICU_ADMISSION').withColumn('INFANT_NICU_ADMISSION_RECODE',
                                                               df_births.INFANT_NICU_ADMISSION) \
        .replace(to_replace=['Y', 'N', 'U'], value=['1', '0', '0'], subset='INFANT_NICU_ADMISSION_RECODE')
    df1.show(3)

    # 将births_transformed中所有含有YN的列都改为01的列
    exprs_YNU = [rec_integer(x, functions.lit('YNU')).alias(x) if x in YNU_cols else x for x in
                 births_transformed.columns]
    df2 = births_transformed.select(exprs_YNU)
    births_transformed.select(YNU_cols[-5:]).show(3)  # 打印YNU_cols列用来检查有没有问题
    return df2


# 5.3.1了解你的数据
def learn2(df2):
    numeric_cols = ['MOTHER_AGE_YEARS', 'FATHER_COMBINED_AGE', 'CIG_BEFORE', 'CIG_1_TRI', 'CIG_2_TRI', 'CIG_3_TRI',
                    'MOTHER_HEIGHT_IN', 'MOTHER_PRE_WEIGHT', 'MOTHER_DELIVERY_WEIGHT', 'MOTHER_WEIGHT_GAIN']
    # 标准的将df转换成为rdd的写法
    rdd1 = df2.select(numeric_cols).rdd.map(lambda x: [i for i in x])
    # 调用MLlib运算rdd的各种信息
    mllib_stats = stat.Statistics.colStats(rdd1)
    # 打印出rdd没列的均值、平方差
    for col, m, v in zip(numeric_cols, mllib_stats.mean(), mllib_stats.variance()):
        print('{0}: \t{1:.2f} \t {2:.2f}'.format(col, m, numpy.sqrt(v)))
    # 将在df2.columns但不在numeric_cols的列名提取出来，放在一个list中
    categorical_cols = [e for e in df2.columns if e not in numeric_cols]
    # 将numeric_cols的列从df2中提取出来，并转换成为标准rdd
    categorical_rdd = df2.select(categorical_cols).rdd.map(lambda x: [i1 for i1 in x])

    print("DF3:")
    df3 = ss.createDataFrame(data=categorical_rdd, schema=categorical_cols)
    df3.show(3)

    print('len(categorical_cols):', len(categorical_cols))
    for i, col in enumerate(categorical_cols):
        # #groupBy(lambda row: row[i])的原因是df3的每列数值都是只有0 1两类，之前的0 1都已经转换、筛出来了
        # groupBy用于将1 0分成2组，map用于计算每个数值的总数（用len计算）
        agg = categorical_rdd.groupBy(lambda row: row[i]).map(lambda row: (row[0], len(row[1])))
        # 按agg中的第二维度el[1]对数据进行sorted升序排列，并展示出来。sorted为python函数
        print('类别频率统计:', i, col, sorted(agg.collect(), key=lambda el: el[1], reverse=True))

    return rdd1, numeric_cols, df2


# 数据相关性计算：相关系数
def learn3(numeric_rdd, numeric_cols, df2):
    # 计算相关系数，调用的是MLlib。
    corrs = stat.Statistics.corr(numeric_rdd)
    # print('相关系数：', len(corrs), len(numeric_rdd.collect()), (corrs > 0.5))
    # 只将corrs > 0.5的筛选出来
    for i, el in enumerate(corrs > 0.5):  # el：[False False  True  True  True  True False False False False]
        # 如果corrs > 0.5且不是corrs[i][i]（因为相关系数是两个不同的数之间的相关性，不能是两个相同数的相关性），则记录相关系数值
        # ==改为is则报错。可以直接将if e == True改为if e
        correlated = [(numeric_cols[j], corrs[i][j]) for j, e in enumerate(el) if e == True and j != i]
        if len(correlated) > 0:  # 如果correlated中有内容，则:
            for e in correlated:
                print('相关系数-->', '{0}---to---{1}: {2:.2f}'.format(numeric_cols[i], e[0], e[1]))

    # 看到相关系数的结果后，将相关性强的列去掉，只保留一列即可：
    list_features_to_keep = ['INFANT_ALIVE_AT_REPORT', 'BIRTH_PLACE', 'MOTHER_AGE_YEARS', 'FATHER_COMBINED_AGE', 'CIG_1_TRI',
                        'MOTHER_HEIGHT_IN', 'MOTHER_PRE_WEIGHT', 'DIABETES_PRE', 'DIABETES_GEST', 'HYP_TENS_PRE',
                        'HYP_TENS_GEST', 'PREV_BIRTH_PRETERM']
    for i in df2.schema.names:
        if i in list_features_to_keep:
            pass
        else:print('Not in list_features_to_keep')


if __name__ == '__main__':
    rdd1, numeric_cols, df2 = learn2(learn1())
    learn3(rdd1, numeric_cols, df2)
