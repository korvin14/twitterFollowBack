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

import networkx as nx

# 1st: log regr implementation
# 2nd: random forest
class UsersPair():
    def __init__(self, id1, id2):
        self.id2 = id2
        self.id1 = id1
        self.commonFriends = 0
        
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

# up = UsersPair(4413, 81817)

# start = time.time()

friends = None

with open("graph_cb.txt") as f:
    friends = f.readlines()
    
friendsProcessed = []

for x in friends:
    friendsProcessed.append(x.split(" "))
 
friendsFinal = calcFriends(friendsProcessed)

graph = nx.DiGraph()

print len(friendsFinal)

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
    
    x.commonFriends = count
    
print graph.number_of_edges()
print "graph.number_of_edges()"

for x in friendsFinal:
    print x.commonFriends
# 
# for x in friendsFinal:
#     counter = 0
#     id1Friends = getFriends(x.id1, friendsFinal1)
#     id2Friends = getFriends(x.id2, friendsFinal1)
#     for y in id1Friends:
#         if y in id2Friends:
#             counter += 1
#     x.commonFriends = counter      
# 
# features = []
# answers = []
# 
# print time.time() - start
#  
# for x in friendsFinal:
#     answers.append(friendsFinal[x])
#     tmp = []
#     if not (x in outTweets):
#         tmp.append(0)
#     else:
#         tmp.append(outTweets[x])
#         
#     backPair = UsersPair(x.id2, x.id1)
#     if not (backPair in outTweets):
#         tmp.append(0)
#     else:
#         tmp.append(outTweets[backPair])
#         
#     features.append(tmp)
# 
# # for x in outTweets:
# #     if (x in friendsFinal):
# #         features.append([outTweets.get(x)])
# #         answers.append(friendsFinal[x])
#           
# """LOGISTIC REGR"""  
# regr = LogisticRegression(solver="sag")
# # 
# x_train, x_test, y_train, y_test = train_test_split(features, answers, test_size=0.2, random_state=999)
# # 
# print "xtr and xtest"
# print len(x_train)
# print len(x_test)
# regr.fit(x_train, y_train)
# # 
# predicted_hui = regr.predict(x_test)
# predicted_hui_train = regr.predict(x_train)
# 
# reg_proba_test = regr.predict_proba(x_test)
# reg_proba_tr = regr.predict_proba(x_train)
# 
# acc_hui = accuracy_score(y_test, predicted_hui)
# loss_reg_test = log_loss(y_test, reg_proba_test)
# acc_hui_train = accuracy_score(y_train, predicted_hui_train)
# loss_reg_tr = log_loss(y_train, reg_proba_tr)
# print "acc_test {}\n acc_train {}\n test loss {}\n tr loss {}".format(acc_hui, acc_hui_train, loss_reg_test, loss_reg_tr)
# 
# """RANDOM FOREST"""
# # rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5)
# # rf = rf.fit(x_train, y_train)
# # predicted_hui_rf = rf.predict(x_test)
# # trained_hui_rf = rf.predict(x_train)
# # test_acc = accuracy_score(y_test, predicted_hui_rf)
# # # frst_log_loss_1 = log_loss(y_test, frst_probalities_1)
# # tr_acc = accuracy_score(y_train, trained_hui_rf)
# # # frst_log_loss_1_tr = log_loss(y_train, frst_probalities_1_tr)
# # print "acc_test {}\n acc_train {}".format(test_acc, tr_acc)
# 
# print "data access time ", time.time() - start
# 


