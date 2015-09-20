"""
   测试数据来源，Berkeley大学公开课程
   个人实现，仅供交流学习
"""

import sys
import os
from test_helper import Test

baseDir = os.path.join('data')
inputPath = os.path.join('small')

ratingsFilename = os.path.join(baseDir, inputPath, 'ratings.dat.gz')
moviesFilename = os.path.join(baseDir, inputPath, 'movies.dat')

numPartitions = 2
rawRatings = sc.textFile(ratingsFilename).repartition(numPartitions)
rawMovies = sc.textFile(moviesFilename)

def get_ratings_tuple(entry):
    items = entry.split('::')
    return int(items[0]), int(items[1]), float(items[2])


def get_movie_tuple(entry):
    items = entry.split('::')
    return int(items[0]), items[1]


ratingsRDD = rawRatings.map(get_ratings_tuple).cache()
moviesRDD = rawMovies.map(get_movie_tuple).cache()

ratingsCount = ratingsRDD.count()
moviesCount = moviesRDD.count()

print 'There are %s ratings and %s movies in the datasets' % (ratingsCount, moviesCount)
print 'Ratings: %s' % ratingsRDD.take(3)
print 'Movies: %s' % moviesRDD.take(3)

assert ratingsCount == 487650
assert moviesCount == 3883
assert moviesRDD.filter(lambda (id, title): title == 'Toy Story (1995)').count() == 1
assert (ratingsRDD.takeOrdered(1, key=lambda (user, movie, rating): movie)
        == [(1, 1, 5.0)])

tmp1 = [(1, u'alpha'), (2, u'alpha'), (2, u'beta'), (3, u'alpha'), (1, u'epsilon'), (1, u'delta')]
tmp2 = [(1, u'delta'), (2, u'alpha'), (2, u'beta'), (3, u'alpha'), (1, u'epsilon'), (1, u'alpha')]

oneRDD = sc.parallelize(tmp1)
twoRDD = sc.parallelize(tmp2)
oneSorted = oneRDD.sortByKey(True).collect()
twoSorted = twoRDD.sortByKey(True).collect()
print oneSorted
print twoSorted
assert set(oneSorted) == set(twoSorted)     
assert twoSorted[0][0] < twoSorted.pop()[0] 
assert oneSorted[0:2] != twoSorted[0:2]     


def sortFunction(tuple):
    key = unicode('%.3f' % tuple[0])
    value = tuple[1]
    return (key + ' ' + value)


print oneRDD.sortBy(sortFunction, True).collect()
print twoRDD.sortBy(sortFunction, True).collect()

oneSorted1 = oneRDD.takeOrdered(oneRDD.count(),key=sortFunction)
twoSorted1 = twoRDD.takeOrdered(twoRDD.count(),key=sortFunction)
print 'one is %s' % oneSorted1
print 'two is %s' % twoSorted1
assert oneSorted1 == twoSorted1


# 基础推荐算法

def getCountsAndAverages(IDandRatingsTuple):
    tempTuple = (len(IDandRatingsTuple[1]), sum(IDandRatingsTuple[1])/float(len(IDandRatingsTuple[1])))
    return IDandRatingsTuple[0],tempTuple

Test.assertEquals(getCountsAndAverages((1, (1, 2, 3, 4))), (1, (4, 2.5)),
                            'incorrect getCountsAndAverages() with integer list')
Test.assertEquals(getCountsAndAverages((100, (10.0, 20.0, 30.0))), (100, (3, 20.0)),
                            'incorrect getCountsAndAverages() with float list')
Test.assertEquals(getCountsAndAverages((110, xrange(20))), (110, (20, 9.5)),
                            'incorrect getCountsAndAverages() with xrange')

movieIDsWithRatingsRDD = (ratingsRDD
                          .map(lambda x:(x[1],x[2]))
                          .groupByKey())
print 'movieIDsWithRatingsRDD: %s\n' % movieIDsWithRatingsRDD.take(3)

movieIDsWithAvgRatingsRDD = movieIDsWithRatingsRDD.map(lambda x:getCountsAndAverages(x))
print 'movieIDsWithAvgRatingsRDD: %s\n' % movieIDsWithAvgRatingsRDD.take(3)

movieNameWithAvgRatingsRDD = (moviesRDD
                              .join(movieIDsWithAvgRatingsRDD)
                              .map(lambda x:(x[1][1][1],x[1][0],x[1][1][0])))
print 'movieNameWithAvgRatingsRDD: %s\n' % movieNameWithAvgRatingsRDD.take(3)

Test.assertEquals(movieIDsWithRatingsRDD.count(), 3615,
                'incorrect movieIDsWithRatingsRDD.count() (expected 3615)')
movieIDsWithRatingsTakeOrdered = movieIDsWithRatingsRDD.takeOrdered(3)
Test.assertTrue(movieIDsWithRatingsTakeOrdered[0][0] == 1 and
                len(list(movieIDsWithRatingsTakeOrdered[0][1])) == 993,
                'incorrect count of ratings for movieIDsWithRatingsTakeOrdered[0] (expected 993)')
Test.assertTrue(movieIDsWithRatingsTakeOrdered[1][0] == 2 and
                len(list(movieIDsWithRatingsTakeOrdered[1][1])) == 332,
                'incorrect count of ratings for movieIDsWithRatingsTakeOrdered[1] (expected 332)')
Test.assertTrue(movieIDsWithRatingsTakeOrdered[2][0] == 3 and
                len(list(movieIDsWithRatingsTakeOrdered[2][1])) == 299,
                'incorrect count of ratings for movieIDsWithRatingsTakeOrdered[2] (expected 299)')

Test.assertEquals(movieIDsWithAvgRatingsRDD.count(), 3615,
                'incorrect movieIDsWithAvgRatingsRDD.count() (expected 3615)')
Test.assertEquals(movieIDsWithAvgRatingsRDD.takeOrdered(3),
                [(1, (993, 4.145015105740181)), (2, (332, 3.174698795180723)),
                 (3, (299, 3.0468227424749164))],
                'incorrect movieIDsWithAvgRatingsRDD.takeOrdered(3)')

Test.assertEquals(movieNameWithAvgRatingsRDD.count(), 3615,
                'incorrect movieNameWithAvgRatingsRDD.count() (expected 3615)')
Test.assertEquals(movieNameWithAvgRatingsRDD.takeOrdered(3),
                [(1.0, u'Autopsy (Macchie Solari) (1975)', 1), (1.0, u'Better Living (1998)', 1),
                 (1.0, u'Big Squeeze, The (1996)', 3)],
                 'incorrect movieNameWithAvgRatingsRDD.takeOrdered(3)')


# 高评分用户量筛选

movieLimitedAndSortedByRatingRDD = (movieNameWithAvgRatingsRDD
                                    .filter(lambda x:x[2]>500)
                                    .sortBy(sortFunction, False))
print 'Movies with highest ratings: %s' % movieLimitedAndSortedByRatingRDD.take(20)


# In[18]:

# TEST Movies with Highest Average Ratings and more than 500 Reviews (1c)

Test.assertEquals(movieLimitedAndSortedByRatingRDD.count(), 194,
                'incorrect movieLimitedAndSortedByRatingRDD.count()')
Test.assertEquals(movieLimitedAndSortedByRatingRDD.take(20),
              [(4.5349264705882355, u'Shawshank Redemption, The (1994)', 1088),
               (4.515798462852263, u"Schindler's List (1993)", 1171),
               (4.512893982808023, u'Godfather, The (1972)', 1047),
               (4.510460251046025, u'Raiders of the Lost Ark (1981)', 1195),
               (4.505415162454874, u'Usual Suspects, The (1995)', 831),
               (4.457256461232604, u'Rear Window (1954)', 503),
               (4.45468509984639, u'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)', 651),
               (4.43953006219765, u'Star Wars: Episode IV - A New Hope (1977)', 1447),
               (4.4, u'Sixth Sense, The (1999)', 1110), (4.394285714285714, u'North by Northwest (1959)', 700),
               (4.379506641366224, u'Citizen Kane (1941)', 527), (4.375, u'Casablanca (1942)', 776),
               (4.363975155279503, u'Godfather: Part II, The (1974)', 805),
               (4.358816276202219, u"One Flew Over the Cuckoo's Nest (1975)", 811),
               (4.358173076923077, u'Silence of the Lambs, The (1991)', 1248),
               (4.335826477187734, u'Saving Private Ryan (1998)', 1337),
               (4.326241134751773, u'Chinatown (1974)', 564),
               (4.325383304940375, u'Life Is Beautiful (La Vita \ufffd bella) (1997)', 587),
               (4.324110671936759, u'Monty Python and the Holy Grail (1974)', 759),
               (4.3096, u'Matrix, The (1999)', 1250)], 'incorrect sortedByRatingRDD.take(20)')



trainingRDD, validationRDD, testRDD = ratingsRDD.randomSplit([6, 2, 2], seed=0L)

print 'Training: %s, validation: %s, test: %s\n' % (trainingRDD.count(),
                                                    validationRDD.count(),
                                                    testRDD.count())
print trainingRDD.take(3)
print validationRDD.take(3)
print testRDD.take(3)

assert trainingRDD.count() == 292716
assert validationRDD.count() == 96902
assert testRDD.count() == 98032

assert trainingRDD.filter(lambda t: t == (1, 914, 3.0)).count() == 1
assert trainingRDD.filter(lambda t: t == (1, 2355, 5.0)).count() == 1
assert trainingRDD.filter(lambda t: t == (1, 595, 5.0)).count() == 1

assert validationRDD.filter(lambda t: t == (1, 1287, 5.0)).count() == 1
assert validationRDD.filter(lambda t: t == (1, 594, 4.0)).count() == 1
assert validationRDD.filter(lambda t: t == (1, 1270, 5.0)).count() == 1

assert testRDD.filter(lambda t: t == (1, 1193, 5.0)).count() == 1
assert testRDD.filter(lambda t: t == (1, 2398, 4.0)).count() == 1
assert testRDD.filter(lambda t: t == (1, 1035, 5.0)).count() == 1


import math

#RMSE误差指标
def computeError(predictedRDD, actualRDD):

    predictedReformattedRDD = predictedRDD.map(lambda x:((x[0],x[1]),x[2]))
    
    actualReformattedRDD = actualRDD.map(lambda x:((x[0],x[1]),x[2]))

    squaredErrorsRDD = (predictedReformattedRDD
                        .join(actualReformattedRDD)
                        .map(lambda x:(x[0],pow((x[1][0]-x[1][1]),2))))
    print squaredErrorsRDD.collect()
    
    totalError = squaredErrorsRDD.map(lambda x:x[1]).reduce(lambda x,y:x+y)

    numRatings = squaredErrorsRDD.count()

    return math.sqrt(float(totalError)/float(numRatings))

testPredicted = sc.parallelize([
    (1, 1, 5),
    (1, 2, 3),
    (1, 3, 4),
    (2, 1, 3),
    (2, 2, 2),
    (2, 3, 4)])
testActual = sc.parallelize([
     (1, 2, 3),
     (1, 3, 5),
     (2, 1, 5),
     (2, 2, 1)])
testPredicted2 = sc.parallelize([
     (2, 2, 5),
     (1, 2, 5)])
testError = computeError(testPredicted, testActual)
print 'Error for test dataset (should be 1.22474487139): %s' % testError

testError2 = computeError(testPredicted2, testActual)
print 'Error for test dataset2 (should be 3.16227766017): %s' % testError2

testError3 = computeError(testActual, testActual)
print 'Error for testActual dataset (should be 0.0): %s' % testError3

Test.assertTrue(abs(testError - 1.22474487139) < 0.00000001,
                'incorrect testError (expected 1.22474487139)')
Test.assertTrue(abs(testError2 - 3.16227766017) < 0.00000001,
                'incorrect testError2 result (expected 3.16227766017)')
Test.assertTrue(abs(testError3 - 0.0) < 0.00000001,
                'incorrect testActual result (expected 0.0)')


# MLLIB ALS算法填充评分矩阵
from pyspark.mllib.recommendation import ALS

validationForPredictRDD = validationRDD.map(lambda x:(x[0],x[1]))

seed = 5L
iterations = 5
regularizationParameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.03

minError = float('inf')
bestRank = -1
bestIteration = -1
for rank in ranks:
    model = ALS.train(trainingRDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
    predictedRatingsRDD = model.predictAll(validationForPredictRDD)
    error = computeError(predictedRatingsRDD, validationRDD)
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < minError:
        minError = error
        bestRank = rank

print 'The best model was trained with rank %s' % bestRank

Test.assertEquals(trainingRDD.getNumPartitions(), 2,
                  'incorrect number of partitions for trainingRDD (expected 2)')
Test.assertEquals(validationForPredictRDD.count(), 96902,
                  'incorrect size for validationForPredictRDD (expected 96902)')
Test.assertEquals(validationForPredictRDD.filter(lambda t: t == (1, 1907)).count(), 1,
                  'incorrect content for validationForPredictRDD')
Test.assertTrue(abs(errors[0] - 0.883710109497) < tolerance, 'incorrect errors[0]')
Test.assertTrue(abs(errors[1] - 0.878486305621) < tolerance, 'incorrect errors[1]')
Test.assertTrue(abs(errors[2] - 0.876832795659) < tolerance, 'incorrect errors[2]')

#模型测试
myModel = ALS.train(trainingRDD, bestRank, seed=seed, iterations=iterations,lambda_=regularizationParameter)
testForPredictingRDD = testRDD.map(lambda x:(x[0],x[1]))
predictedTestRDD = myModel.predictAll(testForPredictingRDD)

testRMSE = computeError(testRDD, predictedTestRDD)

print 'The model had a RMSE on the test set of %s' % testRMSE

Test.assertTrue(abs(testRMSE - 0.87809838344) < tolerance, 'incorrect testRMSE')


#对照测试
trainingAvgRating = trainingRDD.map(lambda x:x[2]).reduce(lambda x,y:x+y)/float(trainingRDD.count())
print 'The average rating for movies in the training set is %s' % trainingAvgRating

testForAvgRDD = testRDD.map(lambda x:(x[0],x[1],trainingAvgRating))
testAvgRMSE = computeError(testRDD, testForAvgRDD)
print 'The RMSE on the average set is %s' % testAvgRMSE

Test.assertTrue(abs(trainingAvgRating - 3.57409571052) < 0.000001,
                'incorrect trainingAvgRating (expected 3.57409571052)')
Test.assertTrue(abs(testAvgRMSE - 1.12036693569) < 0.000001,
                'incorrect testAvgRMSE (expected 1.12036693569)')


print 'Most rated movies:'
print '(average rating, movie name, number of reviews)'
for ratingsTuple in movieLimitedAndSortedByRatingRDD.take(50):
    print ratingsTuple


#测试预测结果
myUserID = 0

myRatedMovies = [
     (myUserID, 260, 5),
     (myUserID, 354, 4),
     (myUserID, 642, 3),
     (myUserID, 175, 5),
     (myUserID, 716, 2),
     (myUserID, 212, 4),
     (myUserID, 3910, 4),
     (myUserID, 3910, 5),
     (myUserID, 2601, 3),
     (myUserID, 2160, 2),
     (myUserID, 1460, 3),
     (myUserID, 1254, 4),
     (myUserID, 2740, 2),
     (myUserID, 1104, 3),
     (myUserID, 890, 5),
    ]
myRatingsRDD = sc.parallelize(myRatedMovies)
print 'My movie ratings: %s' % myRatingsRDD.take(10)

trainingWithMyRatingsRDD = trainingRDD.union(myRatingsRDD)

print ('The training dataset now has %s more entries than the original training dataset' %
       (trainingWithMyRatingsRDD.count() - trainingRDD.count()))
assert (trainingWithMyRatingsRDD.count() - trainingRDD.count()) == myRatingsRDD.count()

myRatingsModel = ALS.train(trainingWithMyRatingsRDD, bestRank, seed=seed, iterations=iterations,lambda_=regularizationParameter)

predictedTestMyRatingsRDD = myRatingsModel.predictAll(testForPredictingRDD)
testRMSEMyRatings = computeError(testRDD, predictedTestMyRatingsRDD)
print 'The model had a RMSE on the test set of %s' % testRMSEMyRatings

myUnratedMoviesRDD = (moviesRDD
                      .map(lambda x:(myUserID, x[0]))
                      .filter(lambda x:x[1] not in [x[1] for x in myRatedMovies]))

predictedRatingsRDD = myRatingsModel.predictAll(myUnratedMoviesRDD)


movieCountsRDD = movieIDsWithAvgRatingsRDD.map(lambda x:(x[0],x[1][0]))

predictedRDD = predictedRatingsRDD.map(lambda x:(x[1],x[2]))

predictedWithCountsRDD  = (predictedRDD
                           .join(movieCountsRDD))

ratingsWithNamesRDD = (predictedWithCountsRDD
                       .map(lambda x:(x[1][0],x[0],x[1][1]))
                       .filter(lambda x:x[2]>75))

predictedHighestRatedMovies = ratingsWithNamesRDD.takeOrdered(20, key=lambda x: -x[0])
print ('My highest rated movies as predicted (for movies with more than 75 reviews):\n%s' %
        '\n'.join(map(str, predictedHighestRatedMovies)))

