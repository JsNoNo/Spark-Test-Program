"""
   测试数据来源：berkeley大学公开课程
   个人实现，供交流学习 
"""

import re
DATAFILE_PATTERN = '^(.+),"(.+)",(.*),(.*),(.*)'

#去除引号
def removeQuotes(s):
    return ''.join(i for i in s if i!='"')

#记录处理
def parseDatafileLine(datafileLine):   
    match = re.search(DATAFILE_PATTERN, datafileLine)
    if match is None:
        print 'Invalid datafile line: %s' % datafileLine
        return (datafileLine, -1)
    elif match.group(1) == '"id"':
        print 'Header datafile line: %s' % datafileLine
        return (datafileLine, 0)
    else:
        product = '%s %s %s' % (match.group(2), match.group(3), match.group(4))
        return ((removeQuotes(match.group(1)), product), 1)

import sys
import os
from test_helper import Test

baseDir = os.path.join('data')
inputPath = os.path.join('testData')

GOOGLE_PATH = 'Google.csv'
GOOGLE_SMALL_PATH = 'Google_small.csv'
AMAZON_PATH = 'Amazon.csv'
AMAZON_SMALL_PATH = 'Amazon_small.csv'
GOLD_STANDARD_PATH = 'Amazon_Google_perfectMapping.csv'
STOPWORDS_PATH = 'stopwords.txt'

def parseData(filename):
    return (sc
            .textFile(filename, 4, 0)
            .map(parseDatafileLine)
            .cache())

def loadData(path):
    filename = os.path.join(baseDir, inputPath, path)
    raw = parseData(filename).cache()
    failed = (raw
              .filter(lambda s: s[1] == -1)
              .map(lambda s: s[0]))
    for line in failed.take(10):
        print '%s - Invalid datafile line: %s' % (path, line)
    valid = (raw
             .filter(lambda s: s[1] == 1)
             .map(lambda s: s[0])
             .cache())
    print '%s - Read %d lines, successfully parsed %d lines, failed to parse %d lines' % (path,
                                                                                        raw.count(),
                                                                                        valid.count(),
                                                                                        failed.count())
    assert failed.count() == 0
    assert raw.count() == (valid.count() + 1)
    return valid

googleSmall = loadData(GOOGLE_SMALL_PATH)
google = loadData(GOOGLE_PATH)
amazonSmall = loadData(AMAZON_SMALL_PATH)
amazon = loadData(AMAZON_PATH)

for line in googleSmall.take(3):
    print 'google: %s: %s\n' % (line[0], line[1])

for line in amazonSmall.take(3):
    print 'amazon: %s: %s\n' % (line[0], line[1])


# 词袋模型处理
quickbrownfox = 'A quick brown fox jumps over the lazy dog.'
split_regex = r'\W+'

def simpleTokenize(string):
    return filter(None,re.split(split_regex,string.lower()))

print simpleTokenize(quickbrownfox)

Test.assertEquals(simpleTokenize(quickbrownfox),
                  ['a','quick','brown','fox','jumps','over','the','lazy','dog'],
                  'simpleTokenize should handle sample text')
Test.assertEquals(simpleTokenize(' '), [], 'simpleTokenize should handle empty string')
Test.assertEquals(simpleTokenize('!!!!123A/456_B/789C.123A'), ['123a','456_b','789c','123a'],
                  'simpleTokenize should handle puntuations and lowercase result')
Test.assertEquals(simpleTokenize('fox fox'), ['fox', 'fox'],
                  'simpleTokenize should not remove duplicates')


# 去除停用词
stopfile = os.path.join(baseDir, inputPath, STOPWORDS_PATH)
stopwords = set(sc.textFile(stopfile).collect())
print 'These are the stopwords: %s' % stopwords

def tokenize(string):
    return [string for string in simpleTokenize(string) if string not in stopwords]

print tokenize(quickbrownfox)

Test.assertEquals(tokenize("Why a the?"), [], 'tokenize should remove all stopwords')
Test.assertEquals(tokenize("Being at the_?"), ['the_'], 'tokenize should handle non-stopwords')
Test.assertEquals(tokenize(quickbrownfox), ['quick','brown','fox','jumps','lazy','dog'],
                    'tokenize should handle sample text')

amazonRecToToken = amazonSmall.map(lambda (k,v): (k,tokenize(v)))
googleRecToToken = googleSmall.map(lambda (k,v): (k,tokenize(v)))

def countTokens(vendorRDD):
    return vendorRDD.mapValues(lambda v:len(v)).map(lambda (k,v): v).reduce(lambda x,y:x+y)
totalTokens = countTokens(amazonRecToToken) + countTokens(googleRecToToken)
print 'There are %s tokens in the combined datasets' % totalTokens

Test.assertEquals(totalTokens, 22520, 'incorrect totalTokens')

#倒排输出
def findBiggestRecord(vendorRDD):
    return vendorRDD.takeOrdered(1, lambda (k,v):-1*len(v))

biggestRecordAmazon = findBiggestRecord(amazonRecToToken)
print 'The Amazon record with ID "%s" has the most tokens (%s)' % (biggestRecordAmazon[0][0],
                                                                   len(biggestRecordAmazon[0][1]))

Test.assertEquals(biggestRecordAmazon[0][0], 'b000o24l3q', 'incorrect biggestRecordAmazon')
Test.assertEquals(len(biggestRecordAmazon[0][1]), 1547, 'incorrect len for biggestRecordAmazon')

#词频计算tf
def tf(tokens):
    termFrequent = {}
    for term in tokens:
        if term in termFrequent.keys():
            termFrequent[term] +=1.0
        else:
            termFrequent[term] = 1.0
    length = len(tokens)
    for key in termFrequent.keys():
        termFrequent[key] /= float(length)
    return termFrequent

print tf(tokenize(quickbrownfox)) # Should give { 'quick': 0.1666 ... }

tf_test = tf(tokenize(quickbrownfox))
Test.assertEquals(tf_test, {'brown': 0.16666666666666666, 'lazy': 0.16666666666666666,
                             'jumps': 0.16666666666666666, 'fox': 0.16666666666666666,
                             'dog': 0.16666666666666666, 'quick': 0.16666666666666666},
                    'incorrect result for tf on sample text')
tf_test2 = tf(tokenize('one_ one_ two!'))
Test.assertEquals(tf_test2, {'one_': 0.6666666666666666, 'two': 0.3333333333333333},
                    'incorrect result for tf test')

corpusRDD = amazonRecToToken.union(googleRecToToken)

Test.assertEquals(corpusRDD.count(), 400, 'incorrect corpusRDD.count()')


# 逆词频计算
def idfs(corpus):
    N = corpus.count()
    uniqueTokens = corpus.flatMap(lambda (k,v):set(v))
    tokenCountPairTuple = uniqueTokens.map(lambda x:(x,1))
    tokenSumPairTuple = tokenCountPairTuple.reduceByKey(lambda x,y:x+y)
    return (tokenSumPairTuple.map(lambda (k,v):(k,float(N)/v)))

idfsSmall = idfs(amazonRecToToken.union(googleRecToToken))
uniqueTokenCount = idfsSmall.count()

print 'There are %s unique tokens in the small datasets.' % uniqueTokenCount

Test.assertEquals(uniqueTokenCount, 4772, 'incorrect uniqueTokenCount')
tokenSmallestIdf = idfsSmall.takeOrdered(1, lambda s: s[1])[0]
Test.assertEquals(tokenSmallestIdf[0], 'software', 'incorrect smallest IDF token')
Test.assertTrue(abs(tokenSmallestIdf[1] - 4.25531914894) < 0.0000000001,
                'incorrect smallest IDF value')

smallIDFTokens = idfsSmall.takeOrdered(11, lambda s: s[1])
print smallIDFTokens

import matplotlib.pyplot as plt

small_idf_values = idfsSmall.map(lambda s: s[1]).collect()
fig = plt.figure(figsize=(8,3))
plt.hist(small_idf_values, 50, log=True)
pass

#tf-idf计算
def tfidf(tokens, idfs):
    tfs = tf(tokens)
    tfIdfDict = {}
    for key in tfs.keys():
        tfIdfDict[key]=tfs[key]*idfs[key]
    return tfIdfDict

recb000hkgj8k = amazonRecToToken.filter(lambda x: x[0] == 'b000hkgj8k').collect()[0][1]
idfsSmallWeights = idfsSmall.collectAsMap()
rec_b000hkgj8k_weights = tfidf(recb000hkgj8k, idfsSmallWeights)

print 'Amazon record "b000hkgj8k" has tokens and weights:\n%s' % rec_b000hkgj8k_weights


Test.assertEquals(rec_b000hkgj8k_weights,
                   {'autocad': 33.33333333333333, 'autodesk': 8.333333333333332,
                    'courseware': 66.66666666666666, 'psg': 33.33333333333333,
                    '2007': 3.5087719298245617, 'customizing': 16.666666666666664,
                    'interface': 3.0303030303030303}, 'incorrect rec_b000hkgj8k_weights')


import math

#内积
def dotprod(a, b):
    dotProdResult = 0
    unionKeys = [key for key in a.keys() if key in b.keys()]
    for key in unionKeys:
        dotProdResult += a[key] * b[key]
    return dotProdResult

#norm函数
def norm(a):
    normResult = math.sqrt(dotprod(a,a))
    return normResult

#余弦相似性
def cossim(a, b):
    cossimResult = dotprod(a,b)/(norm(a)*norm(b))
    return cossimResult

testVec1 = {'foo': 2, 'bar': 3, 'baz': 5 }
testVec2 = {'foo': 1, 'bar': 0, 'baz': 20 }
dp = dotprod(testVec1, testVec2)
nm = norm(testVec1)
print dp, nm

Test.assertEquals(dp, 102, 'incorrect dp')
Test.assertTrue(abs(nm - 6.16441400297) < 0.0000001, 'incorrrect nm')

def cosineSimilarity(string1, string2, idfsDictionary):
    w1 = tfidf(tokenize(string1),idfsDictionary)
    w2 = tfidf(tokenize(string2),idfsDictionary)
    return cossim(w1, w2)

cossimAdobe = cosineSimilarity('Adobe Photoshop',
                               'Adobe Illustrator',
                               idfsSmallWeights)

print cossimAdobe

Test.assertTrue(abs(cossimAdobe - 0.0577243382163) < 0.0000001, 'incorrect cossimAdobe')

crossSmall = (googleSmall
              .cartesian(amazonSmall)
              .cache())

def computeSimilarity(record):
    googleRec = record[0]
    amazonRec = record[1]
    googleURL = record[0][0]
    amazonID = record[1][0]
    googleValue = record[0][1]
    amazonValue = record[1][1]
    cs = cosineSimilarity(googleValue, amazonValue, idfsSmallWeights)
    return (googleURL, amazonID, cs)

similarities = (crossSmall
                .map(lambda x: computeSimilarity(x))
                .cache())

def similar(amazonID, googleURL):
    return (similarities
            .filter(lambda record: (record[0] == googleURL and record[1] == amazonID))
            .collect()[0][2])

similarityAmazonGoogle = similar('b000o24l3q', 'http://www.google.com/base/feeds/snippets/17242822440574356561')
print 'Requested similarity is %s.' % similarityAmazonGoogle

Test.assertTrue(abs(similarityAmazonGoogle - 0.000303171940451) < 0.0000001,
                'incorrect similarityAmazonGoogle')

def computeSimilarityBroadcast(record):
    googleRec = record[0]
    amazonRec = record[1]
    googleURL = record[0][0]
    amazonID = record[1][0]
    googleValue = record[0][1]
    amazonValue = record[1][1]
    cs = cosineSimilarity(googleValue, amazonValue, idfsSmallBroadcast.value)
    return (googleURL, amazonID, cs)

#对于查询数据，生成广播变量
idfsSmallBroadcast = sc.broadcast(idfsSmallWeights)
similaritiesBroadcast = (crossSmall
                         .map(lambda x: computeSimilarityBroadcast(x))
                         .cache())


def similarBroadcast(amazonID, googleURL):
    return (similaritiesBroadcast
            .filter(lambda record: (record[0] == googleURL and record[1] == amazonID))
            .collect()[0][2])

similarityAmazonGoogleBroadcast = similarBroadcast('b000o24l3q', 'http://www.google.com/base/feeds/snippets/17242822440574356561')
print 'Requested similarity is %s.' % similarityAmazonGoogleBroadcast

from pyspark import Broadcast
Test.assertTrue(isinstance(idfsSmallBroadcast, Broadcast), 'incorrect idfsSmallBroadcast')
Test.assertEquals(len(idfsSmallBroadcast.value), 4772, 'incorrect idfsSmallBroadcast value')
Test.assertTrue(abs(similarityAmazonGoogleBroadcast - 0.000303171940451) < 0.0000001,
                'incorrect similarityAmazonGoogle')

GOLDFILE_PATTERN = '^(.+),(.+)'

def parse_goldfile_line(goldfile_line):
    match = re.search(GOLDFILE_PATTERN, goldfile_line)
    if match is None:
        print 'Invalid goldfile line: %s' % goldfile_line
        return (goldfile_line, -1)
    elif match.group(1) == '"idAmazon"':
        print 'Header datafile line: %s' % goldfile_line
        return (goldfile_line, 0)
    else:
        key = '%s %s' % (removeQuotes(match.group(1)), removeQuotes(match.group(2)))
        return ((key, 'gold'), 1)

goldfile = os.path.join(baseDir, inputPath, GOLD_STANDARD_PATH)
gsRaw = (sc
         .textFile(goldfile)
         .map(parse_goldfile_line)
         .cache())

gsFailed = (gsRaw
            .filter(lambda s: s[1] == -1)
            .map(lambda s: s[0]))
for line in gsFailed.take(10):
    print 'Invalid goldfile line: %s' % line

goldStandard = (gsRaw
                .filter(lambda s: s[1] == 1)
                .map(lambda s: s[0])
                .cache())
print 'Read %d lines, successfully parsed %d lines, failed to parse %d lines' % (gsRaw.count(),
                                                                                 goldStandard.count(),
                                                                                 gsFailed.count())
assert (gsFailed.count() == 0)
assert (gsRaw.count() == (goldStandard.count() + 1))


sims = similaritiesBroadcast.map(lambda (x,y,z):(y+' '+x,z))

trueDupsRDD = (sims
               .join(goldStandard))

trueDupsCount = trueDupsRDD.count()
avgSimDups =  trueDupsRDD.map(lambda x: x[1][0]).reduce(lambda x, y: x+y)/float(trueDupsCount)

nonDupsRDD = (sims
              .subtractByKey(goldStandard))
avgSimNon = nonDupsRDD.map(lambda (k,v):v).reduce(lambda x,y:x+y)/float(nonDupsRDD.count())

print 'There are %s true duplicates.' % trueDupsCount
print 'The average similarity of true duplicates is %s.' % avgSimDups
print 'And for non duplicates, it is %s.' % avgSimNon

Test.assertEquals(trueDupsCount, 146, 'incorrect trueDupsCount')
Test.assertTrue(abs(avgSimDups - 0.264332573435) < 0.0000001, 'incorrect avgSimDups')
Test.assertTrue(abs(avgSimNon - 0.00123476304656) < 0.0000001, 'incorrect avgSimNon')


amazonFullRecToToken = amazon.map(lambda (k,v): (k,tokenize(v)))
googleFullRecToToken = google.map(lambda (k,v): (k,tokenize(v)))

print 'Amazon full dataset is %s products, Google full dataset is %s products' % (amazonFullRecToToken.count(),
                                                                                    googleFullRecToToken.count())

Test.assertEquals(amazonFullRecToToken.count(), 1363, 'incorrect amazonFullRecToToken.count()')
Test.assertEquals(googleFullRecToToken.count(), 3226, 'incorrect googleFullRecToToken.count()')


# 对整个数据集计算处理
fullCorpusRDD = amazonFullRecToToken.union(googleFullRecToToken)
idfsFull = idfs(fullCorpusRDD)
idfsFullCount = idfsFull.count()
print 'There are %s unique tokens in the full datasets.' % idfsFullCount

# Recompute IDFs for full dataset
idfsFullWeights = idfsFull.collectAsMap()
idfsFullBroadcast = sc.broadcast(idfsFullWeights)

# Pre-compute TF-IDF weights.  Build mappings from record ID weight vector.
amazonWeightsRDD = amazonFullRecToToken.map(lambda tokens: (tokens[0], tfidf(tokens[1], idfsFullBroadcast.value)))
googleWeightsRDD = googleFullRecToToken.map(lambda tokens: (tokens[0], tfidf(tokens[1], idfsFullBroadcast.value)))

print 'There are %s Amazon weights and %s Google weights.' % (amazonWeightsRDD.count(),
                                                              googleWeightsRDD.count())


# In[59]:

# TEST Compute IDFs and TF-IDFs for the full datasets (4b)
Test.assertEquals(idfsFullCount, 17078, 'incorrect idfsFullCount')
Test.assertEquals(amazonWeightsRDD.count(), 1363, 'incorrect amazonWeightsRDD.count()')
Test.assertEquals(googleWeightsRDD.count(), 3226, 'incorrect googleWeightsRDD.count()')

amazonNorms = amazonWeightsRDD.map(lambda (a,b): (a, norm(b))).collectAsMap()
amazonNormsBroadcast = sc.broadcast(amazonNorms)
googleNorms = googleWeightsRDD.map(lambda (a,b): (a, norm(b))).collectAsMap()
googleNormsBroadcast = sc.broadcast(googleNorms)

Test.assertTrue(isinstance(amazonNormsBroadcast, Broadcast), 'incorrect amazonNormsBroadcast')
Test.assertEquals(len(amazonNormsBroadcast.value), 1363, 'incorrect amazonNormsBroadcast.value')
Test.assertTrue(isinstance(googleNormsBroadcast, Broadcast), 'incorrect googleNormsBroadcast')
Test.assertEquals(len(googleNormsBroadcast.value), 3226, 'incorrect googleNormsBroadcast.value')


# 反转word-tfidf优化查询，减少计算量
def invert(record):
    pairs = list( map( lambda y : (y,record[0]), record[1].keys() ) )
    return (pairs)

amazonInvPairsRDD = (amazonWeightsRDD
                    .flatMap(lambda x:invert(x))
                    .cache())

googleInvPairsRDD = (googleWeightsRDD
                    .flatMap(lambda x:invert(x))
                    .cache())

print 'There are %s Amazon inverted pairs and %s Google inverted pairs.' % (amazonInvPairsRDD.count(),
                                                                            googleInvPairsRDD.count())

invertedPair = invert((1, {'foo': 2}))
Test.assertEquals(invertedPair[0][1], 1, 'incorrect invert result')
Test.assertEquals(amazonInvPairsRDD.count(), 111387, 'incorrect amazonInvPairsRDD.count()')
Test.assertEquals(googleInvPairsRDD.count(), 77678, 'incorrect googleInvPairsRDD.count()')

def swap(record):
    token = record[0]
    keys = record[1]
    return (keys, token)

commonTokens = (amazonInvPairsRDD
                .join(googleInvPairsRDD)
                .map(lambda x: swap(x))
                .groupByKey()
                .cache())

print 'Found %d common tokens' % commonTokens.count()

Test.assertEquals(commonTokens.count(), 2441100, 'incorrect commonTokens.count()')

amazonWeightsBroadcast = sc.broadcast(amazonWeightsRDD.collectAsMap())
googleWeightsBroadcast = sc.broadcast(googleWeightsRDD.collectAsMap())

def fastCosineSimilarity(record):
    amazonRec = record[0][0]
    googleRec = record[0][1]
    tokens = record[1]
    s = dotprod(amazonWeightsBroadcast.value[amazonRec], googleWeightsBroadcast.value[googleRec])
    value = s/((amazonNormsBroadcast.value[amazonRec])*(googleNormsBroadcast.value[googleRec]))
    key = (amazonRec, googleRec)
    return (key, value)

similaritiesFullRDD = (commonTokens
                       .map(lambda x: fastCosineSimilarity(x))
                       .cache())

print similaritiesFullRDD.count()

similarityTest = similaritiesFullRDD.filter(lambda ((aID, gURL), cs): aID == 'b00005lzly' and gURL == 'http://www.google.com/base/feeds/snippets/13823221823254120257').collect()
Test.assertEquals(len(similarityTest), 1, 'incorrect len(similarityTest)')
Test.assertTrue(abs(similarityTest[0][1] - 4.286548414e-06) < 0.000000000001, 'incorrect similarityTest fastCosineSimilarity')
Test.assertEquals(similaritiesFullRDD.count(), 2441100, 'incorrect similaritiesFullRDD.count()')


#数据分析

simsFullRDD = similaritiesFullRDD.map(lambda x: ("%s %s" % (x[0][0], x[0][1]), x[1]))
assert (simsFullRDD.count() == 2441100)

simsFullValuesRDD = (simsFullRDD
                     .map(lambda x: x[1])
                     .cache())
assert (simsFullValuesRDD.count() == 2441100)

def gs_value(record):
    if (record[1][1] is None):
        return 0
    else:
        return record[1][1]

trueDupSimsRDD = (goldStandard
                  .leftOuterJoin(simsFullRDD)
                  .map(gs_value)
                  .cache())
print 'There are %s true duplicates.' % trueDupSimsRDD.count()
assert(trueDupSimsRDD.count() == 1300)

from pyspark.accumulators import AccumulatorParam
class VectorAccumulatorParam(AccumulatorParam):
    # Initialize the VectorAccumulator to 0
    def zero(self, value):
        return [0] * len(value)

    # Add two VectorAccumulator variables
    def addInPlace(self, val1, val2):
        for i in xrange(len(val1)):
            val1[i] += val2[i]
        return val1

def set_bit(x, value, length):
    bits = []
    for y in xrange(length):
        if (x == y):
          bits.append(value)
        else:
          bits.append(0)
    return bits

BINS = 101
nthresholds = 100
def bin(similarity):
    return int(similarity * nthresholds)

zeros = [0] * BINS
fpCounts = sc.accumulator(zeros, VectorAccumulatorParam())

def add_element(score):
    global fpCounts
    b = bin(score)
    fpCounts += set_bit(b, 1, BINS)

simsFullValuesRDD.foreach(add_element)

def sub_element(score):
    global fpCounts
    b = bin(score)
    fpCounts += set_bit(b, -1, BINS)

trueDupSimsRDD.foreach(sub_element)

def falsepos(threshold):
    fpList = fpCounts.value
    return sum([fpList[b] for b in range(0, BINS) if float(b) / nthresholds >= threshold])

def falseneg(threshold):
    return trueDupSimsRDD.filter(lambda x: x < threshold).count()

def truepos(threshold):
    return trueDupSimsRDD.count() - falsenegDict[threshold]


# Precision Recall F-score评测指标

def precision(threshold):
    tp = trueposDict[threshold]
    return float(tp) / (tp + falseposDict[threshold])

def recall(threshold):
    tp = trueposDict[threshold]
    return float(tp) / (tp + falsenegDict[threshold])

def fmeasure(threshold):
    r = recall(threshold)
    p = precision(threshold)
    return 2 * r * p / (r + p)


thresholds = [float(n) / nthresholds for n in range(0, nthresholds)]
falseposDict = dict([(t, falsepos(t)) for t in thresholds])
falsenegDict = dict([(t, falseneg(t)) for t in thresholds])
trueposDict = dict([(t, truepos(t)) for t in thresholds])

precisions = [precision(t) for t in thresholds]
recalls = [recall(t) for t in thresholds]
fmeasures = [fmeasure(t) for t in thresholds]

print precisions[0], fmeasures[0]
assert (abs(precisions[0] - 0.000532546802671) < 0.0000001)
assert (abs(fmeasures[0] - 0.00106452669505) < 0.0000001)


fig = plt.figure()
plt.plot(thresholds, precisions)
plt.plot(thresholds, recalls)
plt.plot(thresholds, fmeasures)
plt.legend(['Precision', 'Recall', 'F-measure'])
pass

