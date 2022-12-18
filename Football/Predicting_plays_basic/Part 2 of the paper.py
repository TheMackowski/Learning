# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:11:20 2021

@author: 15869
"""
# In[]:
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

import pandas as pd
# In[]:

lr_score =[]
abc_score = []
gbt_score = []
rf_score = []
random_score = []

for outer_ind  in range(4,11,2):
    df = pd.read_excel('final_clean_df_' +str(outer_ind) + '_.xlsx')
    
    output_col = df[0]
    numeric_col = ['O_playing','D_playing','Temp','Week']
    catagorical_col = ['Inside','Pricipitation','oca','dca']
    
    for item in numeric_col:
        temp_max = df[item].max()
        df[item] = df.apply(lambda x: x[item]/temp_max,axis= 1)
    for item in catagorical_col:
        df[item] = df[item].astype('category')
    
    clean_df = df[[#'O_playing',
                   'D_playing',
                   #'Temp',
                   
                   'rb_rank','qb_rank','wr_rank',
                   'rb_playing','qb_playing',
                   
                   'wr_playing'
                   ,'HOME_AWAY'
                  # ,'Week','Inside','Pricipitation'
                   #,'oca','dca'
                   ,0]]
    
    
    
    X_train, X_test, y_train, y_test = train_test_split( clean_df.iloc[:,:-1],clean_df.iloc[:,-1] , test_size=0.20, random_state=42)
    
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    pred_y = lr.predict(X_test)
    cm_lr = metrics.confusion_matrix(y_test, pred_y)
    scorelr = lr.score(X_test, y_test)
    
    RF = RandomForestClassifier(max_depth=3, random_state=42,n_estimators = 10)
    RF.fit(X_train, y_train)
    pred_y = RF.predict(X_test)
    cm_RF = metrics.confusion_matrix(y_test, pred_y)
    scoreRF = RF.score(X_test, y_test)
    

    abc = AdaBoostClassifier()
    abc.fit(X_train, y_train)
    pred_y = abc.predict(X_test)
    cm_abc = metrics.confusion_matrix(y_test, pred_y)
    scoreabc = abc.score(X_test, y_test)
    
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    pred_y = gbc.predict(X_test)
    cm_gbc= metrics.confusion_matrix(y_test, pred_y)
    scoregbc = gbc.score(X_test, y_test)
    
    lr_score.append(scorelr)
    abc_score.append(scoreabc)
    gbt_score.append(scoregbc)
    rf_score.append(scoreRF)
    random_score.append(1/outer_ind)

# In[]:
plt.scatter(range(4,11,2),lr_score)
plt.scatter(range(4,11,2),abc_score)
plt.scatter(range(4,11,2),gbt_score)
plt.scatter(range(4,11,2),rf_score)

plt.plot(range(4,11,2),random_score)
plt.legend(['random','rf','gbt','abc','lr'])




# In[]:
