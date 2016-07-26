import sklearn as sk
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
# import networkx as nx
import time
from sklearn.cross_validation import train_test_split
# from sklearn.decomposition.tests.test_nmf import random_state
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.ensemble.forest import RandomForestClassifier, \
    RandomForestRegressor
from sklearn.linear_model.logistic import LogisticRegression
# 1st: log regr implementation
# 2nd: random forest
class UsersPair():
    def __init__(self, id1, id2):
        self.id2 = id2
        self.id1 = id1
        
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
            friendsFinal[nextPairStraight] = 0  
    
    return friendsFinal

start = time.time()

tweets = None

with open("interaction_list_all.txt") as f:
    tweets = f.readlines()

tweetsProcessed = []

for x in tweets:
    tweetsProcessed.append(x.split(" "))
 
outTweets = calcTweets(tweetsProcessed)

# up = UsersPair(4413, 81817)

# start = time.time()

friends = None

with open("graph_cb.txt") as f:
    friends = f.readlines()
    
friendsProcessed = []

for x in friends:
    friendsProcessed.append(x.split(" "))
 
friendsFinal = calcFriends(friendsProcessed)

features = []
answers = []

 
for x in friendsFinal:
    answers.append(friendsFinal[x])
    if not (x in outTweets):
        features.append([0])
    else:
        features.append([outTweets[x]])
      
"""LOGISTIC REGR"""  
regr = LogisticRegression(solver="sag")
# 
x_train, x_test, y_train, y_test = train_test_split(features, answers, test_size=0.6, random_state=999)
# 
print "xtr and xtest"
print len(x_train)
print len(x_test)
regr.fit(x_train, y_train)
# 
predicted_hui = regr.predict(x_test)
predicted_hui_train = regr.predict(x_train)

reg_proba_test = regr.predict_proba(x_test)
reg_proba_tr = regr.predict_proba(x_train)

acc_hui = accuracy_score(y_test, predicted_hui)
loss_reg_test = log_loss(y_test, reg_proba_test)
acc_hui_train = accuracy_score(y_train, predicted_hui_train)
loss_reg_tr = log_loss(y_train, reg_proba_tr)
print "acc_test {}\n acc_train {}\n test loss {}\n tr loss {}".format(acc_hui, acc_hui_train, loss_reg_test, loss_reg_tr)

"""RANDOM FOREST"""
# rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5)
# rf = rf.fit(x_train, y_train)
# predicted_hui_rf = rf.predict(x_test)
# trained_hui_rf = rf.predict(x_train)
# test_acc = accuracy_score(y_test, predicted_hui_rf)
# # frst_log_loss_1 = log_loss(y_test, frst_probalities_1)
# tr_acc = accuracy_score(y_train, trained_hui_rf)
# # frst_log_loss_1_tr = log_loss(y_train, frst_probalities_1_tr)
# print "acc_test {}\n acc_train {}".format(test_acc, tr_acc)

print "data access time ", time.time() - start



