import sklearn as sk
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
# import networkx as nx
import time
from sklearn.cross_validation import train_test_split
# from sklearn.decomposition.tests.test_nmf import random_state
from sklearn.metrics.classification import accuracy_score, log_loss
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


i = 0

for x in friendsFinal:
    answers.append(friendsFinal[x])
    if not (x in outTweets):
        features.append([0])
    else:
        features.append([outTweets[x]])
        
regr = LogisticRegressionCV(cv=5, solver="liblinear", class_weight=None, n_jobs=4, random_state=999)

x_train, x_test, y_train, y_test = train_test_split(features, answers, test_size=0.4, random_state=999)

regr.fit(x_train, y_train)

predicted_hui = regr.predict(x_test)
predicted_hui_train = regr.predict(x_train)

acc_hui = accuracy_score(y_test, predicted_hui)
# reg_log_loss_C1 = log_loss(y_test, reg_probalities_C1)
acc_hui_train = accuracy_score(y_train, predicted_hui_train)
# reg_log_loss_C1_tr = log_loss(y_train, reg_probalities_C1_tr)
print "acc_test {}\n acc_train {}".format(acc_hui, acc_hui_train)

print "data access time ", time.time() - start



