import sklearn as sk
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
import numpy as np
# import networkx as nx
import time
from sklearn.cross_validation import train_test_split, KFold
# from sklearn.decomposition.tests.test_nmf import random_state
from sklearn.metrics.classification import accuracy_score, log_loss, \
    precision_score, recall_score, f1_score
from sklearn.ensemble.forest import RandomForestClassifier, \
    RandomForestRegressor
from sklearn.linear_model.logistic import LogisticRegression

import networkx as nx

# 1st: log lr implementation
# 2nd: random forest
class UsersPair():
    def __init__(self, id1, id2):
        self.id2 = id2
        self.id1 = id1
        self.commonFriends = 0
        self.friendOfFriends = 0
        self.id1Friends = 0
        self.id2Friends = 0
        
        
    def __eq__(self, other):
        return (self.id1 == other.id1 and self.id2 == other.id2)
    
    def __hash__(self):
        return hash((self.id1, self.id2))
    
    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)
   
def calcTweets(tweetsProcessed):
    outTweets = {}
    
    curPair = UsersPair(int(tweetsProcessed[0][0]), int(tweetsProcessed[0][1]))

    outTweets[curPair] = 0
 
    for x in tweetsProcessed:    
        nextPair = UsersPair(int(x[0]), int(x[1]))
        if nextPair in outTweets:
            outTweets[nextPair] += 1
        else:
            outTweets[nextPair] = 1
    
    return outTweets

def calcFriends(friendsProcessed):
    friendsFinal = {}
        
    for x in friendsProcessed:
        nextPairStraight = UsersPair(int(x[0]), int(x[1]))
        nextPairBack = UsersPair(int(x[1]), int(x[0]))
        if nextPairBack in friendsFinal:
            friendsFinal[nextPairStraight] = 1
            friendsFinal[nextPairBack] = 1
        else:
            friendsFinal[nextPairStraight] = -1  
    
    return friendsFinal

def getFriends(id, friendsFinal):
    result = []
    for x in friendsFinal:
        if (x.id1 == id):
            result.append(x.id2)
    
    return result

start = time.time()

tweets = None

with open("interaction_list_all.txt") as f:
    tweets = f.readlines()

tweetsProcessed = []

for x in tweets:
    tweetsProcessed.append(x.split(" "))
 
outTweets = calcTweets(tweetsProcessed)

friends = None

with open("graph_cb.txt") as f:
    friends = f.readlines()
    
friendsProcessed = []

for x in friends:
    friendsProcessed.append(x.split(" "))
 
friendsFinal = calcFriends(friendsProcessed)

graph = nx.DiGraph()

for x in friendsFinal:
    graph.add_node(x.id1)
    graph.add_node(x.id2)
    graph.add_edge(x.id1, x.id2)

for x in friendsFinal:
    grandChildren = []
    count = 0
    children = graph.neighbors(x.id1)
    for y in children:
        grandChildren.extend(graph.neighbors(y))
         
    for y in grandChildren:
        if y == x.id2:
            count += 1
    
    x.friendOfFriends = count
    count = 0
            
    children1 = graph.neighbors(x.id1)
    children2 = graph.neighbors(x.id2)
    
    for y in children1:
        if y in children2:
            count += 1
    
    x.commonFriends = count
    x.id1Friends = len(children1)
    x.id2Friends = len(children2)    
    
    
print "number of edges: ", graph.number_of_edges()

features = []
answers = []
 
print "mapped friends of friends in", time.time() - start, "s"
start = time.time()
  
for x in friendsFinal:
    answers.append(friendsFinal[x])
    tmp = []
    if not (x in outTweets):
        tmp.append(0)
    else:
        tmp.append(outTweets[x])
           
    backPair = UsersPair(x.id2, x.id1)
    if not (backPair in outTweets):
        tmp.append(0)
    else:
        tmp.append(outTweets[backPair])
    tmp.append(x.friendOfFriends)
    tmp.append(x.commonFriends)
    tmp.append(x.id1Friends)    
    tmp.append(x.id2Friends) 
    features.append(tmp)
    
for x in xrange(len(features)):
    features[x].append(answers[x])
    

print "mapped features", time.time() - start
start = time.time()
           
"""LOGISTIC REGR"""  
# 
# x_train, x_test, y_train, y_test = train_test_split(features, answers, test_size=0.2)
lr = LogisticRegression(C=0.00001, solver="sag", n_jobs=4)

kf = KFold(len(features), n_folds=5, shuffle=True)
# lrScores = []
lrAvgPrecision = 0.0
lrAvgRecall = 0.0
lrAvgF1 = 0.0
for train, test in kf:
    y_train = []
    x_train = []
    for i in train:
        y_train.append(features[i][6])
        tmp = [features[i][0], features[i][1], features[i][2], features[i][3], features[i][4], features[i][5]]
        x_train.append(tmp)
        
    y_test = []
    x_test = []  
    for i in test:
        y_test.append(features[i][6])
        tmp = [features[i][0], features[i][1], features[i][2], features[i][3], features[i][4], features[i][5]]
        x_test.append(tmp)
        
    lr.fit(x_train, y_train)
    lrPredTest = lr.predict(x_test)
    lrPrecisionTest = precision_score(y_test, lrPredTest)
    lrRecallTest = recall_score(y_test, lrPredTest)
    lrF1Test = f1_score(y_test, lrPredTest)
    lrAvgPrecision += lrPrecisionTest
    lrAvgRecall += lrRecallTest
    lrAvgF1 += lrF1Test

print "log reg completed in ", time.time() - start, " s"
print "lr:\n Precision {}\n Recall {}\n F1 {}\n".format(lrAvgPrecision / 5, lrAvgRecall / 5, lrAvgF1 / 5)
  

start = time.time()
"""RANDOM FOREST"""
rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5)

rfAvgPrecision = 0.0
rfAvgRecall = 0.0
rfAvgF1 = 0.0
for train, test in kf:
    y_train = []
    x_train = []
    for i in train:
        y_train.append(features[i][6])
        tmp = [features[i][0], features[i][1], features[i][2], features[i][3], features[i][4], features[i][5]]
        x_train.append(tmp)
        
    y_test = []
    x_test = []  
    for i in test:
        y_test.append(features[i][6])
        tmp = [features[i][0], features[i][1], features[i][2], features[i][3], features[i][4], features[i][5]]
        x_test.append(tmp)
       
    rf.fit(x_train, y_train)
    rfPredTest = rf.predict(x_test)
    rfPrecisionTest = precision_score(y_test, rfPredTest)
    rfRecallTest = recall_score(y_test, rfPredTest)
    rfF1Test = f1_score(y_test, rfPredTest)
    rfAvgPrecision += rfPrecisionTest
    rfAvgRecall += rfRecallTest
    rfAvgF1 += rfF1Test

print "RF completed in ", time.time() - start, " s"
print "rf:\n Precision {}\n Recall {}\n F1 {}\n".format(rfAvgPrecision / 5, rfAvgRecall / 5, rfAvgF1 / 5)

 


