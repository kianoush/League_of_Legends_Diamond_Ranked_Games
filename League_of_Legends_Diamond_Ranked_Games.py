"""
Context
League of Legends is a MOBA (multiplayer online battle arena) where 2 teams (blue and red) face off. There are 3 lanes, a jungle, and 5 roles. The goal is to take down the enemy Nexus to win the game.

Content
This dataset contains the first 10min. stats of approx. 10k ranked games (SOLO QUEUE) from a high ELO (DIAMOND I to MASTER). Players have roughly the same level.

Each game is unique. The gameId can help you to fetch more attributes from the Riot API.

There are 19 features per team (38 in total) collected after 10min in-game. This includes kills, deaths, gold, experience, level… It's up to you to do some feature engineering to get more insights.

The column blueWins is the target value (the value we are trying to predict). A value of 1 means the blue team has won. 0 otherwise.

So far I know, there is no missing value.

Glossary
Warding totem: An item that a player can put on the map to reveal the nearby area. Very useful for map/objectives control.
Minions: NPC that belong to both teams. They give gold when killed by players.
Jungle minions: NPC that belong to NO TEAM. They give gold and buffs when killed by players.
Elite monsters: Monsters with high hp/damage that give a massive bonus (gold/XP/stats) when killed by a team.
Dragons: Elite monster which gives team bonus when killed. The 4th dragon killed by a team gives a massive stats bonus. The 5th dragon (Elder Dragon) offers a huge advantage to the team.
Herald: Elite monster which gives stats bonus when killed by the player. It helps to push a lane and destroys structures.
Towers: Structures you have to destroy to reach the enemy Nexus. They give gold.
Level: Champion level. Start at 1. Max is 18.


https://www.kaggle.com/bobbyscience/league-of-legends-diamond-ranked-games-10-min

"""
import time, datetime
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, cv


#print(os.listdir())
raw_data = pd.read_csv('high_diamond_ranked_10min.csv')
#raw_data.info()
raw_data.sort_values('blueTotalGold', axis = 0, ascending = True,
                 inplace = True, na_position ='first')
df = raw_data.iloc[:, 2:40]
lable = raw_data.iloc[:, 1]

"""
List of important feature
"""

list_of_important_feature = ['blueWardsDestroyed','blueKills', 'blueDeaths', 'blueAssists', 'blueEliteMonsters',
                             'blueDragons','blueTowersDestroyed','blueAvgLevel', 'blueTotalMinionsKilled', 'redDeaths']
df_2 = pd.DataFrame()
for titel in list_of_important_feature:
    df_2[titel] = raw_data[titel]


list_of_important_feature01 = ['blueTotalExperience','blueGoldDiff', 'redTotalGold', 'redGoldDiff', 'redExperienceDiff']
df_3 = pd.DataFrame()
for titel in list_of_important_feature01:
    df_3[titel] = raw_data[titel]
"""
Data split
"""
x_train, x_test, y_train, y_test = train_test_split(df, lable, test_size=0.2, shuffle=True, random_state=12)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=12)


def fit_ml_algo(algo, X_train, y_train, cv):
    # One Pass
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)

    # Cross Validation
    train_pred = model_selection.cross_val_predict(algo,
                                                   X_train,
                                                   y_train,
                                                   cv=cv,
                                                   n_jobs=-1)
    # Cross-validation accuracy metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

    return train_pred, acc, acc_cv

# Logistic Regression
start_time = time.time()
train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(),
                                                               x_train,
                                                               y_train,
                                                                    10)
log_time = (time.time() - start_time)
print("Logistic Regression Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))
print()


# k-Nearest Neighbours
start_time = time.time()
train_pred_knn, acc_knn, acc_cv_knn = fit_ml_algo(KNeighborsClassifier(n_neighbors=3),
                                                  x_train,
                                                  y_train,
                                                  10)
knn_time = (time.time() - start_time)
print("KNN Accuracy: %s" % acc_knn)
print("Accuracy CV 10-Fold: %s" % acc_cv_knn)
print("Running Time: %s" % datetime.timedelta(seconds=knn_time))
print()

# Linear SVC
start_time = time.time()
train_pred_svc, acc_linear_svc, acc_cv_linear_svc = fit_ml_algo(LinearSVC(),
                                                                x_train,
                                                                y_train,
                                                                10)
linear_svc_time = (time.time() - start_time)
print("SVC Accuracy: %s" % acc_linear_svc)
print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)
print("Running Time: %s" % datetime.timedelta(seconds=linear_svc_time))
print()







"""
MLP classification
"""
x_train = torch.tensor(x_train.values).float()
x_test = torch.tensor(x_test.values).float()
x_valid = torch.tensor(x_valid.values).float()

y_train = torch.tensor(y_train.values)
y_test = torch.tensor(y_test.values)
y_valid = torch.tensor(y_valid.values)

"""
Model
"""
feature_num = x_train.shape[1]
class_num = 2
hidden_layer = 10


model = torch.nn.Sequential(nn.Linear(feature_num, hidden_layer),
                            nn.ReLU(),
                            nn.Linear(hidden_layer, class_num),
                            nn.Sigmoid())
"""
Loss
"""

loss = torch.nn.CrossEntropyLoss()


"""
Optim
"""
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

"""
Train
"""

epochs = 200

train_sample_num = torch.tensor(x_train.shape[0])
test_sample_num = torch.tensor(x_test.shape[0])
valid_sample_num = torch.tensor(x_valid.shape[0])

# for epoch in range(epochs):
#
#     optimizer.zero_grad()
#     yp = model(x_train)
#     loss_1 = loss(yp, y_train)
#     num_corrects = torch.sum(torch.max(yp, 1)[1] == y_train)
#
#     loss_1.backward()
#     optimizer.step()
#
#
#     train_acc = num_corrects.float()/ float(train_sample_num)
#
#     yp_valid = model(x_valid)
#     valid_corrects = torch.sum(torch.max(yp_valid, 1)[1] == y_valid)
#     valid_acc = valid_corrects.float() / float(valid_sample_num)
#
#
#     print('epoch_num= ', epoch, 'Loss= ', loss_1.item(), 'train_acc= ', train_acc.item(), 'valida_acc= ', valid_acc.item())







print('END!')