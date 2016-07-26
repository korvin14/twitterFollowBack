# HW6: Twitter followback prediction
https://github.com/korvin14/twitterFollowBack
## Required soft:
* python 2.7 
* Networkx 1.11
* Sklearn 0.18.1

## How to run:
_Considering, all libs are installed:_
python dbscan.py
__time - around 3min__ 

## Features
So, since you know the input data format, we will come down to features:
* Feat1: how many times id1 @ id2
* Feat2: how many times id2 @ id1
* Feat3: how many friends of a friends id1 has
* Feat4: how many common friends id1 and id2 have

_Comment to feat 3: we counted how many friends of id1's friends follow id2. Usually, similar kind of recommendation algorithm is used in russian social network for advising interesting friend profiles._

## Algo
We used standard nested lists to store data. We also used networkx to calculate common friends and friends of friends.

While we were tuning the prediction algos, we tried two classifiers:
* Logistic regression
* Random forest classifier

The first one showed worse results, compared to Random forest. We started learning from one feature and when we added 3 more, the results improved in 15% from the original results.

Logistic regression showed bad results, due to overfitting in test set(recall is 99%).

## Results
Overal, best derived results(Random Forest):
* Precision - 0.7652
* Recall    - 0.6945
* F1        - 0.7281

_results file are enclosed in git and on server in final.txt_

__P.S. Did you use friends of friends feature in your followback prediciton?__
