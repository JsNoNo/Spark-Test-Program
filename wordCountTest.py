"""
   测试数据来源 Berkeley大学公开课程wordcount测试数据集
   个人实现，仅供交流学习
"""

# 测试RDD

wordsList = ['cat', 'elephant', 'rat', 'rat', 'cat']
wordsRDD = sc.parallelize(wordsList, 4)
# Print out the type of wordsRDD
print type(wordsRDD)


# 记录处理 输入参数类型：string，输出参数类型：string
def makePlural(word):
    return word + 's'

print makePlural('cat')

# 测试
from test_helper import Test

Test.assertEquals(makePlural('rat'), 'rats', 'incorrect result: makePlural does not add an s')


# 对数据并行化应用函数
pluralRDD = wordsRDD.map(makePlural)
print pluralRDD.collect()

Test.assertEquals(pluralRDD.collect(), ['cats', 'elephants', 'rats', 'rats', 'cats'],
                  'incorrect values for pluralRDD')

pluralLambdaRDD = wordsRDD.map(lambda x : x+'s')
print pluralLambdaRDD.collect()

Test.assertEquals(pluralLambdaRDD.collect(), ['cats', 'elephants', 'rats', 'rats', 'cats'],
                  'incorrect values for pluralLambdaRDD (1d)')


# 计算单词长度
pluralLengths = (pluralRDD
                 .map(lambda x : len(x))
                 .collect())
print pluralLengths

Test.assertEquals(pluralLengths, [4, 9, 4, 4, 4],
                  'incorrect values for pluralLengths')


# 并行构建元组
wordPairs = wordsRDD.map(lambda x:(x,1))
print wordPairs.collect()

Test.assertEquals(wordPairs.collect(),
                  [('cat', 1), ('elephant', 1), ('rat', 1), ('rat', 1), ('cat', 1)],
                  'incorrect value for wordPairs')


# 计数，应用transform算子
wordsGrouped = wordPairs.groupByKey().mapValues(lambda x:list(x))
for key, value in wordsGrouped.collect():
    print '{0}: {1}'.format(key, list(value))

Test.assertEquals(sorted(wordsGrouped.mapValues(lambda x: list(x)).collect()),
                  [('cat', [1, 1]), ('elephant', [1]), ('rat', [1, 1])],
                  'incorrect value for wordsGrouped')

wordCountsGrouped = wordsGrouped.map(lambda (k,v):(k,sum(v)))
print wordCountsGrouped.collect()

Test.assertEquals(sorted(wordCountsGrouped.collect()),
                  [('cat', 2), ('elephant', 1), ('rat', 2)],
                  'incorrect value for wordCountsGrouped')

wordCounts = wordPairs.reduceByKey(lambda x,y:x+y)
print wordCounts.collect()

Test.assertEquals(sorted(wordCounts.collect()), [('cat', 2), ('elephant', 1), ('rat', 2)],
                  'incorrect value for wordCounts')

wordCountsCollected = (wordsRDD
                       .map(lambda x:(x,1))
                       .reduceByKey(lambda x,y:x+y)
                       .collect())
print wordCountsCollected

Test.assertEquals(sorted(wordCountsCollected), [('cat', 2), ('elephant', 1), ('rat', 2)],
                  'incorrect value for wordCountsCollected')


#输出词集合统计平均次数
uniqueWords = len(wordCountsCollected)
print uniqueWords

Test.assertEquals(uniqueWords, 3, 'incorrect count of uniqueWords')

from operator import add
totalCount = (wordCounts
              .map(lambda (k,v):v)
              .reduce(add))
average = totalCount / float(uniqueWords)
print totalCount
print round(average, 2)

Test.assertEquals(round(average, 2), 1.67, 'incorrect value of average')


# 参数类型：RDD 输出元组(word, count)
def wordCount(wordListRDD):
    wordCounts = wordListRDD.map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y)
    return wordCounts
print wordCount(wordsRDD).collect()

Test.assertEquals(sorted(wordCount(wordsRDD).collect()),
                  [('cat', 2), ('elephant', 1), ('rat', 2)],
                  'incorrect definition for wordCount function')


# 应用正则化处理
import re
import string

#去除标点
def removePunctuation(text):
    pattern = re.compile('[%s]' % string.punctuation)
    return re.sub(pattern,'',text.lower().strip())
print removePunctuation('Hi, you!')
print removePunctuation(' No under_score!')

Test.assertEquals(removePunctuation(" The Elephant's 4 cats. "),
                  'the elephants 4 cats',
                  'incorrect definition for removePunctuation function')

import os.path
baseDir = os.path.join('data')
inputPath = os.path.join('shakespeare.txt')
fileName = os.path.join(baseDir, inputPath)

shakespeareRDD = (sc
                  .textFile(fileName, 8)
                  .map(removePunctuation))
print '\n'.join(shakespeareRDD
                .zipWithIndex()  # to (line, lineNum)
                .map(lambda (l, num): '{0}: {1}'.format(num, l))  # to 'lineNum: line'
                .take(15))

shakespeareWordsRDD = shakespeareRDD.flatMap(lambda x:x.split(' '))
shakespeareWordCount = shakespeareWordsRDD.count()
print shakespeareWordsRDD.top(5)
print shakespeareWordCount

Test.assertTrue(shakespeareWordCount == 927631 or shakespeareWordCount == 928908,
                'incorrect value for shakespeareWordCount')
Test.assertEquals(shakespeareWordsRDD.top(5),
                  [u'zwaggerd', u'zounds', u'zounds', u'zounds', u'zounds'],
                  'incorrect value for shakespeareWordsRDD')

shakeWordsRDD = shakespeareWordsRDD.flatMap(lambda x:x.split(' ')).filter(lambda x:len(x)>0)
shakeWordCount = shakeWordsRDD.count()
print shakeWordCount

Test.assertEquals(shakeWordCount, 882996, 'incorrect value for shakeWordCount')


#计数测试
top15WordsAndCounts = wordCount(shakeWordsRDD).takeOrdered(15,key=lambda x: -x[1])
print '\n'.join(map(lambda (w, c): '{0}: {1}'.format(w, c), top15WordsAndCounts))

Test.assertEquals(top15WordsAndCounts,
                  [(u'the', 27361), (u'and', 26028), (u'i', 20681), (u'to', 19150), (u'of', 17463),
                   (u'a', 14593), (u'you', 13615), (u'my', 12481), (u'in', 10956), (u'that', 10890),
                   (u'is', 9134), (u'not', 8497), (u'with', 7771), (u'me', 7769), (u'it', 7678)],
                  'incorrect value for top15WordsAndCounts')

