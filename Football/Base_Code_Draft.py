# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:01:18 2022

@author: 15869
"""

# In[]:
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from time import sleep

# In[]:
##List of Actual players in 2022 that are planning on playimg 
#set the expectation that these are the players foing to be drafted
list_of_players = pd.read_excel('ListPlayers_2022.xlsx')
running = pd.read_excel('ListPlayers_2022.xlsx', sheet_name = "Running")
recv = pd.read_excel('ListPlayers_2022.xlsx', sheet_name = "Receiving")
passing = pd.read_excel('ListPlayers_2022.xlsx', sheet_name = "Passing")
recv['Att'] = recv['Tgt']
recv['season'] = recv['Year']
running['season'] = running['Year']


# In[]:
#Based on their projected catches and touchdowns collect points that are to be earned.
#Different leagues count pointes differently
#Each possition has a slightly differeny algorithm
df = pd.merge(recv,running, how='outer', on= ['Player','Year'] )
df['Pos_x'].fillna('RB',inplace = True)
df['Pos_y'].fillna('WR',inplace = True)
df2 = pd.merge(passing,running, how='left', on= ['Player','Year'] )
df['points'] = df['Yds_x']/10 + df['Yds_y']/10 + df['TD_x']*6 + df['TD_y']*6 + df['Rec']*.5
df2['points'] = df2['Yds_x']*.04 + 4 *df2['TD_x'] - 2 *df2['Int'] + df2['Yds_y']/10 + df2['TD_y']*6
df_rb = df[df['Pos_x'] == 'RB' ].sort_values('points',ascending=False)#.reset_index(inplace = True)
df_wr= df[df['Pos_y'] == 'WR' ].sort_values('points',ascending=False)#.reset_index(inplace = True)

df_te= df[df['Pos_y'] == 'TE' ].sort_values('points',ascending=False)#.reset_index(inplace = True)
df_qb = df2[df2['Pos_y'] == 'QB' ].sort_values('points',ascending=False)
df_rb['rank'] = df_rb.groupby('Year')['points'].rank(ascending = False)
df_qb['rank'] = df_qb.groupby('Year')['points'].rank(ascending = False)

#df[f'{method}_rank'] = df.groupby('group')['value'].rank(method)
df_wr['rank'] = df_wr.groupby('Year')['points'].rank(ascending = False)
df_te['rank'] = df_te.groupby('Year')['points'].rank(ascending = False)
# In[]:
#RB can run and catch passes, all of them can do any of it
df_rb['Rk'] = df_rb['rank']
df_rb['G'] = df_rb['G_x']
df_rb['Att'] = df_rb['Att_x'] + df_rb['Att_y']
df_rb['Yds'] = df_rb['Yds_x'] + df_rb['Yds_y']
df_rb['TD'] = df_rb['TD_x'] + df_rb['TD_y']

df_wr['Rk'] = df_wr['rank']
df_wr['G'] = df_wr['G_y']
df_wr['Att'] = df_wr['Att_x'] + df_wr['Att_y']
df_wr['Yds'] = df_wr['Yds_x'] + df_wr['Yds_y']
df_wr['TD'] = df_wr['TD_x'] + df_wr['TD_y']

df_te['Rk'] = df_te['rank']
df_te['G'] = df_te['G_y']
df_te['Att'] = df_te['Att_x'] + df_te['Att_y']
df_te['Yds'] = df_te['Yds_x'] + df_te['Yds_y']
df_te['TD'] = df_te['TD_x'] + df_te['TD_y']

df_qb['Rk'] = df_qb['rank']
df_qb['G'] = df_qb['G_y']
df_qb['Att'] = df_qb['Att_x'] + df_qb['Att_y']
df_qb['Yds'] = df_qb['Yds_x']/2.5 + df_qb['Yds_y']
df_qb['TD'] = df_qb['TD_x']/1.5 + df_qb['TD_y'] - df_qb['Int']/3

# In[]:
#Took Three different websites lists as of draft days.

#cur_wr = pd.read_excel('ListPlayers_2022.xlsx', sheet_name = "temp_wr")
#cur_rb = pd.read_excel('ListPlayers_2022.xlsx', sheet_name = "temp_rb")

pff_wr = pd.read_excel('ListPlayers_2022.xlsx', sheet_name = "pff_wr")
pff_rb = pd.read_excel('ListPlayers_2022.xlsx', sheet_name = "pff_rb")

#cbs_wr = pd.read_excel('ListPlayers_2022.xlsx', sheet_name = "cbs_wr")
#cbs_rb = pd.read_excel('ListPlayers_2022.xlsx', sheet_name = "cbs_rb")
# In[]:
#Rank by each year we are looking at 
#This data set looks back at the 2016-2022 (last change of rules)
#Looks at actuals to understand what are the predicited outcomes based on actuals

wdim4 = df_qb[['Rk','G','Att','Yds','TD']].groupby('Rk').agg(['mean' ,'std'])
# In[]:
wdim4[('Points','total')] = wdim4[('Yds','mean')]/10 + 6* wdim4[('TD','mean')]
wdim4[('Points','per_game')] = wdim4[('Points','total')]/wdim4[('G','mean')]
wdim4[('Points','1std_above')] = wdim4[('Points','total')] + (wdim4[('Yds','std')]/10 + 6* wdim4[('TD','std')])
wdim4[('Points','1std_below')] = wdim4[('Points','total')] - (wdim4[('Yds','std')]/10 + 6* wdim4[('TD','std')])

# In[]:
passing_dataset= wdim4.sort_values(('Points','total'),ascending=False).reset_index()
#temp_qb = cur_rb.sort_values('TOT', ascending= False).reset_index()

plt.figure()
plt.plot(passing_dataset[('Points','total')])
plt.plot(passing_dataset[('Points','1std_above')])
plt.plot(passing_dataset[('Points','1std_below')])
#plt.scatter(cur_rb['Rk'],cur_rb['TOT']*16/17)
# plt.scatter(temp_rb.index,temp_rb['TOT']*16/17)
# plt.scatter(pff_rb.index,pff_rb['Pts']*16/17,marker ="x")
# plt.scatter(cbs_rb.index,cbs_rb['FPTS']*16/17,marker ="4")

plt.xlim(0,48)
plt.grid(visible = True)
# In[]:
#hat is the mean anad standard eviation of each rank for each year 
wdim = df_rb[['Rk','G','Att','Yds','TD','Rec']].groupby('Rk').agg(['mean' ,'std'])
# In[]:
wdim[('Points','total')] = wdim[('Yds','mean')]/10 + 6* wdim[('TD','mean')] + wdim[('Rec','mean')]*.5
wdim[('Points','per_game')] = wdim[('Points','total')]/wdim[('G','mean')]
wdim[('Points','1std_above')] = wdim[('Points','total')] + (wdim[('Yds','std')]/10 + 6* wdim[('TD','std')])
wdim[('Points','1std_below')] = wdim[('Points','total')] - (wdim[('Yds','std')]/10 + 6* wdim[('TD','std')])

# In[]:
rushing_dataset= wdim.sort_values(('Points','total'),ascending=False).reset_index()
#temp_rb = cur_rb.sort_values('TOT', ascending= False).reset_index()

plt.figure()
plt.plot(rushing_dataset[('Points','total')])
plt.plot(rushing_dataset[('Points','1std_above')])
plt.plot(rushing_dataset[('Points','1std_below')])
plt.title('Running back past compared to this years projection')
plt.xlabel('ranking')
plt.ylabel('points')
#plt.scatter(cur_rb['Rk'],cur_rb['TOT']*16/17)
#plt.scatter(temp_rb.index,temp_rb['TOT']*16/17)
plt.scatter(pff_rb.index,(pff_rb['Pts'] *16/17) + pff_rb['Rec']*.5,marker ="x")
#plt.scatter(cbs_rb.index,cbs_rb['FPTS']*16/17,marker ="4")

plt.xlim(0,48)
plt.grid(visible = True)


# In[]:
# plt.figure()
# plt.plot(wdim[('Points','1std_above')]-wdim[('Points','1std_below')])
# plt.xlim(0,100)

# In[]:

wdim2 = df_wr[['Rk','G','Att','Yds','TD','Rec']].groupby('Rk').agg(['mean' ,'std'])
# In[]:
wdim2[('Points','total')] = wdim2[('Yds','mean')]/10 + 6* wdim2[('TD','mean')]+ wdim2[('Rec','mean')]*.5
wdim2[('Points','per_game')] = wdim2[('Points','total')]/wdim2[('G','mean')]
wdim2[('Points','1std_above')] = wdim2[('Points','total')] + (wdim2[('Yds','std')]/10 + 6* wdim2[('TD','std')])
wdim2[('Points','1std_below')] = wdim2[('Points','total')] - (wdim2[('Yds','std')]/10 + 6* wdim2[('TD','std')])

# In[]:
wideout_dataset= wdim2.sort_values(('Points','total'),ascending=False).reset_index()
#temp_wr = cur_wr.sort_values('TOT', ascending= False).reset_index()

plt.figure()
plt.plot(wideout_dataset[('Points','total')])
plt.plot(wideout_dataset[('Points','1std_above')])
plt.plot(wideout_dataset[('Points','1std_below')])
plt.title('Wide Receiver past compared to this years projection')
plt.xlabel('ranking')
plt.ylabel('points')
#plt.scatter(cur_wr['Rk'],cur_wr['TOT']*16/17)
#plt.scatter(temp_wr.index,temp_wr['TOT']*16/17)
plt.scatter(pff_wr.index,(pff_wr['Pts'] )*16/17 +.5*pff_wr['Att'],marker ="x")
#plt.scatter(cbs_wr.index,cbs_wr['FPTS']*16/17,marker ="4")


plt.xlim(0,48)
plt.grid(visible = True)


# In[]:

wdim3 = df_te[['Rk','G','Att','Yds','TD','Rec']].groupby('Rk').agg(['mean' ,'std'])
# In[]:
wdim3[('Points','total')] = wdim3[('Yds','mean')]/10 + 6* wdim3[('TD','mean')] + wdim3[('Rec','mean')]*.5
wdim3[('Points','per_game')] = wdim3[('Points','total')]/wdim3[('G','mean')]
wdim3[('Points','1std_above')] = wdim3[('Points','total')] + (wdim3[('Yds','std')]/10 + 6* wdim3[('TD','std')])
wdim3[('Points','1std_below')] = wdim3[('Points','total')] - (wdim3[('Yds','std')]/10 + 6* wdim3[('TD','std')])

# In[]:
tight_dataset= wdim3.sort_values(('Points','total'),ascending=False).reset_index()
#temp_te = cur_te.sort_values('TOT', ascending= False).reset_index()

plt.figure()
plt.plot(tight_dataset[('Points','total')])
plt.plot(tight_dataset[('Points','1std_above')])
plt.plot(tight_dataset[('Points','1std_below')])
#plt.scatter(cur_rb['Rk'],cur_rb['TOT']*16/17)
#plt.scatter(temp_rb.index,temp_rb['TOT']*16/17)
#plt.scatter(pff_rb.index,pff_rb['Pts']*16/17,marker ="x")
#plt.scatter(cbs_rb.index,cbs_rb['FPTS']*16/17,marker ="4")

plt.xlim(0,48)
plt.grid(visible = True)


# In[]:
## TTEst to look at is 1 compared to 2 compared to 3

#when are the drop off in "tiers" when the statisticly signifficant values are not
# so Compare Rank 1 with Rank 2, if they are statistically signifficant compare 
#Rank 1 with 3 and so on untill not statistically significant. 

##Do that untill you have a list of drop off points, Use every other drop off point
#as a Tier structure and every number statistically signifficant to that is a tier. 


import scipy.stats as stats

from scipy.stats import ttest_ind
dict_ranks_rb = {}
temp = df_rb[['Rk','G','Att','Yds','TD','Rec']]
n = 2
p = 2
p_value = 1

for i in tqdm(range(1, 139)):
    
    p_value = 1
    q_value = 0

    while p_value > 0.05:
        
        cat1 = temp[temp['Rk'] == i]
        cat2 = temp[temp['Rk'] == n]
        onee = cat2['Yds']/10 + 6* cat2['TD'] + cat2['Rec']*.5
        twoo = cat1['Yds']/10 + 6* cat1['TD'] + cat1['Rec']*.5
        
        
        
        answer_ = ttest_ind(onee, twoo)
        p_value = answer_[1]
        # print(i)
        # print()
        # print(p_value)
        if p_value <0.05:
            dict_ranks_rb[i] = list(range(i,n))
            # print(i)
            # print("goes to")
            # print(n)
            n=i+2
            
        else:
            n=n+1
    
    while q_value < 0.05 and  i > 2:
        cat1 = temp[temp['Rk'] == i]
        cat2 = temp[temp['Rk'] == p]
        onee = cat2['Yds']/10 + 6* cat2['TD'] +  cat2['Rec']*.5
        twoo = cat1['Yds']/10 + 6* cat1['TD'] + cat1['Rec']*.5
        # print(temp[temp['Rk'] == p])
        # print(i)
        
        answer_ = ttest_ind(onee, twoo)
        q_value = answer_[1]
        # print(i)
        # print(p)
        # print(q_value)
        #print(p_value)
        if q_value >0.05:
            dict_ranks_rb[i].append(p)
            # print(q_value)
            # print("")
            # print(p)
            p = p - 1
            # print(p)
            # print(i)
            # print(i)
            # print("goes to")
            # print(p)
            q_value = 0 
            # print(p)
        else:
           p=i
           # print(q_value)
           q_value = 1
           # q_value = 1
            # n=n+1
# In[]:
i = 2
list_of_tiers_rb = [2]
k = 1
while i < 139:
    #print(i)
    if k == 1:
        i = max(dict_ranks_rb[i])
        k= 0
    else:
        if max(dict_ranks_rb[i]) < 139:
            
            list_of_tiers_rb.append(max(dict_ranks_rb[i]))
            i = max(dict_ranks_rb[i])
            k=1
        else:
            i = max(dict_ranks_rb[i])
            k=1
     
# In[]:
plt.figure()

import numpy as np
# print(np.ones(5)* i)
for i in range(1,len(dict_ranks_rb)):
    ys = np.ones(len(dict_ranks_rb[i]))* i

    plt.plot(dict_ranks_rb[i],ys)
    plt.scatter(list_of_tiers_rb,list_of_tiers_rb)
    plt.grid(visible = True)
    
# In[]:
## TTEst to look at is 1 compared to 2 compared to 3
    
import scipy.stats as stats

from scipy.stats import ttest_ind
dict_ranks_qb = {}
temp = df_qb[['Rk','G','Att','Yds','TD']]
n = 2
p = 2
p_value = 1

for i in range(1, 30):
    p_value = 1
    q_value = 0

    while p_value > 0.05:
        
        cat1 = temp[temp['Rk'] == i]
        cat2 = temp[temp['Rk'] == n]
        onee = cat2['Yds']/10 + 6* cat2['TD']
        twoo = cat1['Yds']/10 + 6* cat1['TD']
        
        
        
        answer_ = ttest_ind(onee, twoo)
        p_value = answer_[1]
        # print(i)
        # print()
        # print(p_value)
        if p_value <0.05:
            dict_ranks_qb[i] = list(range(i,n))
            # print(i)
            # print("goes to")
            # print(n)
            n=i+2
            
        else:
            n=n+1
    
    while q_value < 0.05 and  i > 2:
        cat1 = temp[temp['Rk'] == i]
        cat2 = temp[temp['Rk'] == p]
        onee = cat2['Yds']/10 + 6* cat2['TD']
        twoo = cat1['Yds']/10 + 6* cat1['TD']
        # print(temp[temp['Rk'] == p])
        # print(i)
        
        answer_ = ttest_ind(onee, twoo)
        q_value = answer_[1]
        # print(i)
        # print(p)
        # print(q_value)
        #print(p_value)
        if q_value >0.05:
            dict_ranks_qb[i].append(p)
            # print(q_value)
            # print("")
            # print(p)
            p = p - 1
            # print(p)
            # print(i)
            # print(i)
            # print("goes to")
            # print(p)
            q_value = 0 
            # print(p)
        else:
           p=i
           # print(q_value)
           q_value = 1
           # q_value = 1
            # n=n+1
# In[]:
i = 2
list_of_tiers_qb = [2]
k = 1
while i < 29:
    #print(i)
    if k == 1:
        i = max(dict_ranks_qb[i])
        k= 0
    else:
        if max(dict_ranks_qb[i]) < 29:
            
            list_of_tiers_qb.append(max(dict_ranks_qb[i]))
            i = max(dict_ranks_qb[i])
            k=1
        else:
            i = max(dict_ranks_qb[i])
            k=1
     
# In[]:
plt.figure()

import numpy as np
# print(np.ones(5)* i)
for i in range(1,len(dict_ranks_qb)):
    ys = np.ones(len(dict_ranks_qb[i]))* i

    plt.plot(dict_ranks_qb[i],ys)
    plt.scatter(list_of_tiers_qb,list_of_tiers_qb)
    plt.grid(visible = True)

# In[]:
## TTEst to look at is 1 compared to 2 compared to 3
    
import scipy.stats as stats

from scipy.stats import ttest_ind
dict_ranks_wr = {}
temp = df_wr[['Rk','G','Att','Yds','TD','Rec']]
n = 2
p = 2
p_value = 1

for i in range(1, 100):
    p_value = 1
    q_value = 0

    while p_value > 0.05:
        
        cat1 = temp[temp['Rk'] == i]
        cat2 = temp[temp['Rk'] == n]
        onee = cat2['Yds']/10 + 6* cat2['TD'] + cat2['Rec']*.5
        twoo = cat1['Yds']/10 + 6* cat1['TD']+ cat1['Rec']*.5
        
        
        
        answer_ = ttest_ind(onee, twoo)
        p_value = answer_[1]
        # print(i)
        # print()
        # print(p_value)
        if p_value <0.05:
            dict_ranks_wr[i] = list(range(i,n))
            # print(i)
            # print("goes to")
            # print(n)
            n=i+2
            
        else:
            n=n+1
    
    while q_value < 0.05 and  i > 2:
        cat1 = temp[temp['Rk'] == i]
        cat2 = temp[temp['Rk'] == p]
        onee = cat2['Yds']/10 + 6* cat2['TD']+ cat2['Rec']*.5
        twoo = cat1['Yds']/10 + 6* cat1['TD']+ cat1['Rec']*.5
        # print(temp[temp['Rk'] == p])
        # print(i)
        
        answer_ = ttest_ind(onee, twoo)
        q_value = answer_[1]
        # print(i)
        # print(p)
        # print(q_value)
        #print(p_value)
        if q_value >0.05:
            dict_ranks_wr[i].append(p)
            # print(q_value)
            # print("")
            # print(p)
            p = p - 1
            # print(p)
            # print(i)
            # print(i)
            # print("goes to")
            # print(p)
            q_value = 0 
            # print(p)
        else:
           p=i
           # print(q_value)
           q_value = 1
           # q_value = 1
            # n=n+1
# In[]:
i = 2
list_of_tiers = [2]
k = 1
while i < 100:
    #print(i)
    if k == 1:
        i = max(dict_ranks_wr[i])
        k= 0
    else:
        if max(dict_ranks_wr[i]) < 100:
            
            list_of_tiers.append(max(dict_ranks_wr[i]))
            i = max(dict_ranks_wr[i])
            k=1
        else:
            i = max(dict_ranks_wr[i])
            k=1
       
    
     
# In[]:
import numpy as np
plt.figure()

# print(np.ones(5)* i)
for i in range(1,len(dict_ranks_wr)):
    ys = np.ones(len(dict_ranks_wr[i]))* i

    plt.plot(dict_ranks_wr[i],ys)
    plt.scatter(list_of_tiers,list_of_tiers)
    plt.grid(visible = True)

# In[]:

dict_ranks_te = {}
temp = df_te[['Rk','G','Att','Yds','TD','Rec']]
n = 2
p = 2
p_value = 1

for i in range(1, 5):
    p_value = 1
    q_value = 0

    while p_value > 0.05:
        
        cat1 = temp[temp['Rk'] == i]
        cat2 = temp[temp['Rk'] == n]
        onee = cat2['Yds']/10 + 6* cat2['TD']+ cat2['Rec']*.5
        twoo = cat1['Yds']/10 + 6* cat1['TD']+ cat1['Rec']*.5
        
        
        
        answer_ = ttest_ind(onee, twoo)
        p_value = answer_[1]
        # print(i)
        # print()
        # print(p_value)
        if i ==4:
            
            dict_ranks_te[i] =[5]
            
            p_value = 0
            q_value = 1
        if p_value <0.05:
            dict_ranks_te[i] = list(range(i,n))
            # print(i)
            # print("goes to")
            # print(n)
            n=i+2
            
        else:
            n=n+1
    
    while q_value < 0.05 and  i > 2:
        cat1 = temp[temp['Rk'] == i]
        cat2 = temp[temp['Rk'] == p]
        onee = cat2['Yds']/10 + 6* cat2['TD'] + cat2['Rec']*.5
        twoo = cat1['Yds']/10 + 6* cat1['TD'] + cat1['Rec']*.5
        # print(temp[temp['Rk'] == p])
        # print(i)
        
        answer_ = ttest_ind(onee, twoo)
        q_value = answer_[1]
        # print(i)
        # print(p)
        # print(q_value)
        #print(p_value)
        if q_value >0.05:
            
            dict_ranks_te[i].append(p)
            # print(q_value)
            # print("")
            # print(p)
            p = p - 1
            # print(p)
            # print(i)
            # print(i)
            # print("goes to")
            # print(p)
            q_value = 0 
            # print(p)
        else:
           p=i
           # print(q_value)
           q_value = 1
           # q_value = 1
            # n=n
            
for m in range(6,18):
                dict_ranks_te[4].append(m)
# In[]:
# i = 2
# list_of_tiers_te = [2]
# k = 1
# while i < 2:
    
    
#     if k == 1:
#         i = max(dict_ranks_te[i])
#         k= 0
#     else:
#         list_of_tiers_te.append(max(dict_ranks_te[i]))
#         i = max(dict_ranks_te[i])
#         k=1
        
# list_of_tiers_te.append(max(dict_ranks_te[4]))

# In[]:
qb_tiers = {}
point_tier_qb = {}
i = 1
for item in list_of_tiers_qb:
    print(item)
    qb_tiers[i] = dict_ranks_qb[item]
    point_tier_qb[i] = passing_dataset[('Points','total')][item]
    i = i + 1
plt.figure()

# print(np.ones(5)* i)
for i in range(1,len(qb_tiers)):
    ys = np.ones(len(qb_tiers[i]))* point_tier_qb[i]

    plt.plot(qb_tiers[i],ys)
    #plt.scatter(list_of_tiers,list_of_tiers)
    plt.grid(visible = True)
    plt.title('Quarterback statistically different tiers')
    plt.xlabel('Rank of possition')
    plt.ylabel('Projected Points total')
# In[]:
wr_tiers = {}
point_tier_wr = {}
i = 1
for item in list_of_tiers:
    print(item)
    wr_tiers[i] = dict_ranks_wr[item]
    point_tier_wr[i] = wideout_dataset[('Points','total')][item]
    i = i + 1
plt.figure()

# print(np.ones(5)* i)
for i in range(1,len(wr_tiers)):
    ys = np.ones(len(wr_tiers[i]))* point_tier_wr[i]

    plt.plot(wr_tiers[i],ys)
    #plt.scatter(list_of_tiers,list_of_tiers)
    plt.title('Wide Receiver statistically different tiers')
    plt.xlabel('Rank of possition')
    plt.ylabel('Projected Points total')
    plt.grid(visible = True)
       
# In[]:
rb_tiers = {}
point_tier_rb = {}
i = 1
for item in list_of_tiers_rb:
    #print(item)
    rb_tiers[i] = dict_ranks_rb[item]
    point_tier_rb[i] = rushing_dataset[('Points','total')][item]
    i = i + 1
plt.figure()

# print(np.ones(5)* i)
for i in range(1,len(rb_tiers)):
    ys = np.ones(len(rb_tiers[i]))* point_tier_rb[i]

    plt.plot(rb_tiers[i],ys)
    #plt.scatter(list_of_tiers,list_of_tiers)
    plt.grid(visible = True)
    plt.title('Running back statistically different tiers')
    plt.xlabel('Rank of possition')
    plt.ylabel('Projected Points total')

plt.figure()    
# In[]:
"""this is where we start to unerstand who is in what tier"""

pff_names_rb = {}
max_points_rb = max(point_tier_rb.values())
max_points_wr = max(point_tier_wr.values())

cur_max = max_points_rb

for k, v in point_tier_rb.items():
    
    pff_names_rb[(k,round(v),round(v -cur_max ))] = []
    cur_max = v
cur_max = max_points_rb
j = 1
for i in range(1,len(rb_tiers)):
    for k in rb_tiers[i]:
        pff_names_rb[(i,round(point_tier_rb[i]),round(point_tier_rb[i] - cur_max))].append(pff_rb['Name'][k-1])
    if j == 1:
        pff_names_rb[(i,round(point_tier_rb[i]),round(point_tier_rb[i] - cur_max))].append('RB1')
        j=j+1
    else:
        j=j+1

        del pff_names_rb[(i,round(point_tier_rb[i]),round(point_tier_rb[i] - cur_max))][-1]
    cur_max = point_tier_rb[i]

    
pff_names_wr = {}
cur_max = max_points_wr

for k, v in point_tier_wr.items():
    pff_names_wr[(k,round(v),round(v- cur_max))] = []
    cur_max = v
j=1
cur_max = max_points_wr
for i in range(1,len(wr_tiers)):
    for k in wr_tiers[i]:
        pff_names_wr[(i,round(point_tier_wr[i]),round(point_tier_wr[i] - cur_max))].append(pff_wr['Name'][k-1])
    if j == 1:
        pff_names_wr[(i,round(point_tier_wr[i]),round(point_tier_wr[i] - cur_max))].append('WR1')
        j=j+1
    else:
        del pff_names_wr[(i,round(point_tier_wr[i]),round(point_tier_wr[i] - cur_max))][-1]
    cur_max = point_tier_wr[i]
# In[]:
dict_ranks_te.pop(2)
qb_tiers[1].append(1)


print("Done with Editing - Next is How you want to Draft")
# # In[]:
# People = 12
# teams={}
# points={}
# for i in range(1,People+1):
#     teams[i] = []
#     points[i] = []
    
    
# teams[1]=
# teams[2]=
# teams[3]=
# teams[4]=
# teams[5]=
# teams[6]=
# teams[7]=
# teams[8]=
# teams[9]=
# teams[10]=
# teams[11]=
# teams[12]=
# In[]:

"""Prep
pff_names_rb
pff_names_wr

"""
import random
import copy
team_wr = 3
team_rb = 3
best = {}
results = {}
People = 12
for i in range(1,People+1):
    results[i] = []
    best[i] ={}
# In[]:
j
for i in tqdm(range(1,100000)):
    """ How Do you want to draft """
    
    teams={}
    points={}
    
    for i in range(1,People+1):
        teams[i] = []
        points[i] = []
    rb_left = copy.deepcopy(pff_names_rb)
    wr_left = copy.deepcopy(pff_names_wr)
    te_left = copy.deepcopy(dict_ranks_te)
    qb_left = copy.deepcopy(qb_tiers)
    rb_adjustment = 0 
    wr_adjustment = 0 
    te_adjustment = 0
    qb_adjustment = 0
    QB_PICK = 0
    
    k=1
    snake = 1
    for rounds in range(0,6):
        
        for item in range(1,People+1):
            if snake == 1:
                item = item
            else:
                item = 13- item
            #print(k)
            
            k=k+1
            if rounds == 1 and item < 7: 
                the_pick_is_in = 75

            else:
                the_pick_is_in = random.choice(range(1, 101))
            have_te = 1
            if any(x in ('TE1','TE2','TE3') for x in teams[item]):
                have_te = 0         
            if rounds <2:
                if any(x in ('QB1','QB2') for x in teams[item]):
                    QB_PICK = 0
                else:
                    pick_a_qb = random.choice(range(1, 101))
                    if pick_a_qb > 95 +qb_adjustment:
                        QB_PICK = 1
                    else:
                        QB_PICK = 0
            elif rounds <5:
                if any(x in ('QB1','QB2')  for x in teams[item]):
                    QB_PICK = 0
                else:
                    pick_a_qb = random.choice(range(1, 101))
                    if pick_a_qb > 75 +qb_adjustment:
                        QB_PICK = 1
                    else:
                        QB_PICK = 0
            elif rounds == 9:
                if any(x in ('QB1','QB2')  for x in teams[item]):
                    QB_PICK = 0
                else:
                    pick_a_qb = random.choice(range(1, 101))
                    if pick_a_qb > 0:
                        QB_PICK = 1
                    else:
                        QB_PICK = 0
            else:
                if any(x in ('QB1','QB2')  for x in teams[item]):
                    QB_PICK = 0
                else:
                    pick_a_qb = random.choice(range(1, 101))
                    if pick_a_qb > 60 +qb_adjustment:
                        QB_PICK = 1
                    else:
                        QB_PICK = 0
            if QB_PICK == 1:
                if len(qb_left[list(qb_left.keys())[0]]) > 0:   
                    teams[item].append('QB1')
                    points[item].append(345)
                    del qb_left[list(qb_left.keys())[0]][0]
             #       print('qb1')
                    qb_adjustment = qb_adjustment + 3
                elif len(qb_left[list(qb_left.keys())[1]]) > 0:
                    points[item].append(275)
                    del qb_left[list(qb_left.keys())[1]][0]
                    teams[item].append('QB2')
              #      print('qb2')
                    qb_adjustment = qb_adjustment 
            
            elif (the_pick_is_in < 80 + rb_adjustment + wr_adjustment) and (the_pick_is_in > (12+te_adjustment)*have_te) :
                if len(rb_left[list(rb_left.keys())[0]]) > 0:   
                    teams[item].append('RB1')
                    points[item].append(list(rb_left.keys())[0][1])
                    del rb_left[list(rb_left.keys())[0]][0]
               #     print('rb1')
                    rb_adjustment = rb_adjustment -3
                elif len(rb_left[list(rb_left.keys())[1]]) > 0:
                    points[item].append(list(rb_left.keys())[1][1])
                    del rb_left[list(rb_left.keys())[1]][0]
                    teams[item].append('RB2')
                #    print('rb2')
                    rb_adjustment = rb_adjustment -3
                elif len(rb_left[list(rb_left.keys())[2]]) > 0:
                    points[item].append(list(rb_left.keys())[2][1])
                    del rb_left[list(rb_left.keys())[2]][0]
                    teams[item].append('RB3')
                 #   print('rb3')
                    rb_adjustment = rb_adjustment -3
                elif len(rb_left[list(rb_left.keys())[3]]) > 0:
                    points[item].append(list(rb_left.keys())[3][1])
                    del rb_left[list(rb_left.keys())[3]][0]
                    teams[item].append('RB4')
                  #  print('rb4')
                    rb_adjustment = rb_adjustment -3
                elif len(rb_left[list(rb_left.keys())[4]]) > 0:
                    points[item].append(list(rb_left.keys())[4][1])
                    del rb_left[list(rb_left.keys())[4]][0]
                    teams[item].append('RB5')
               #     print('rb5')
                    rb_adjustment = rb_adjustment -2
                elif len(rb_left[list(rb_left.keys())[5]]) > 0:
                    points[item].append(list(rb_left.keys())[5][1])
                    del rb_left[list(rb_left.keys())[5]][0]
                    teams[item].append('RB6')
                #    print('rb6')
                    rb_adjustment = rb_adjustment -1
                elif len(rb_left[list(rb_left.keys())[6]]) > 0:
                    points[item].append(list(rb_left.keys())[6][1])
                    del rb_left[list(rb_left.keys())[6]][0]
                    teams[item].append('RB7')
                #    print('rb7')
                    rb_adjustment = rb_adjustment -1
                elif len(rb_left[list(rb_left.keys())[7]]) > 0:
                    points[item].append(list(rb_left.keys())[7][1])
                    del rb_left[list(rb_left.keys())[7]][0]
                    teams[item].append('RB8')
                #    print('rb8')
                    rb_adjustment = rb_adjustment -1
                elif len(rb_left[list(rb_left.keys())[8]]) > 0:
                    points[item].append(list(rb_left.keys())[8][1])
                    del rb_left[list(rb_left.keys())[8]][0]
                    teams[item].append('RB9')
                 #   print('rb9')
                    rb_adjustment = rb_adjustment -2
            elif (the_pick_is_in <100) and (the_pick_is_in > 12+te_adjustment):
                if len(wr_left[list(wr_left.keys())[0]]) > 0:
                    points[item].append(list(wr_left.keys())[0][1])
                    del wr_left[list(wr_left.keys())[0]][0]
                    teams[item].append('WR1')
                  #  print('wr1')
                    wr_adjustment = wr_adjustment + 4
                elif len(wr_left[list(wr_left.keys())[1]]) > 0:
                    points[item].append(list(wr_left.keys())[1][1])
                    del wr_left[list(wr_left.keys())[1]][0]
                    teams[item].append('WR2')
                #    print('wr2')
                    wr_adjustment = wr_adjustment + 2
                elif len(wr_left[list(wr_left.keys())[2]]) > 0:
                    points[item].append(list(wr_left.keys())[2][1])
                    del wr_left[list(wr_left.keys())[2]][0]
                    teams[item].append('WR3')
                 #   print('wr3')
                    wr_adjustment = wr_adjustment + 1
                elif len(wr_left[list(wr_left.keys())[3]]) > 0:
                    points[item].append(list(wr_left.keys())[3][1])
                    del wr_left[list(wr_left.keys())[3]][0]
                    teams[item].append('WR4')
                 #   print('wr4')
                    wr_adjustment = wr_adjustment + 0
                elif len(wr_left[list(wr_left.keys())[4]]) > 0:
                    points[item].append(list(wr_left.keys())[4][1])
                    del wr_left[list(wr_left.keys())[4]][0]
                    teams[item].append('WR5')
                  #  print('wr5')
                    wr_adjustment = wr_adjustment + 0
                elif len(wr_left[list(wr_left.keys())[5]]) > 0:
                    points[item].append(list(wr_left.keys())[5][1])
                    del wr_left[list(wr_left.keys())[5]][0]
                    teams[item].append('WR6')
                   # print('wr6')
                    wr_adjustment = wr_adjustment + 0
            else:
                if len(te_left[list(te_left.keys())[0]]) > 0: 
                    points[item].append(150)
                    del te_left[list(te_left.keys())[0]][0]
                    teams[item].append('TE1')
               #     print('te1')
                    te_adjustment = te_adjustment - 2
                elif len(te_left[list(te_left.keys())[1]]) > 0:
                    points[item].append(120)
                    del te_left[list(te_left.keys())[1]][0]
                    teams[item].append('TE2')
                #    print('te2')
                    te_adjustment = te_adjustment - 2
                else :
                    points[item].append(95)
                    del te_left[list(te_left.keys())[2]][0]
                    teams[item].append('TE3')
               #     print('te3')
                    te_adjustment = te_adjustment + 0
            # plt.scatter(k,80 + wr_adjustment + rb_adjustment)
            # plt.xlabel('pick possition')
            # plt.ylabel('Percent chance')
            # plt.title('Chance to pick a Running back')
        if snake == 1:
                snake = 0 
        else:
                snake = 1
        
    
    for k,v in points.items():
        results[k].append((" ".join(str(x) for x in teams[k]),sum(v)))


# In[]:
# Team_Stat = 8



# best ={}
# for item, value in results.items():
#     a = pd.DataFrame(value,columns = ('list','points'))
#     a2 = pd.DataFrame(a.groupby('list')['points'].agg(['mean','count']))
#     a2.sort_values(by = ['mean'],ascending = False,inplace = True)
#     best[item] = a2['mean'][0:26].mean()
#     if item == Team_Stat:
#         output= pd.DataFrame(value,columns = ('list','points'))
#         team_strat = pd.DataFrame(output.groupby('list')['points'].agg(['mean','count']))
#         team_strat.sort_values(by = ['mean'],ascending = False,inplace = True)
#         team_strat.reset_index(inplace =True)
# plt.bar(team_strat['list'][0:3],team_strat['mean'][0:3])

# In[]:

best ={}
std_best = {}
most_likely ={}
std_most_likely ={}
goal = {}
list_of_options = {}
likelyhoods = {}

save_a2s={}
for item, value in results.items():
    a = pd.DataFrame(value,columns = ('list','points'))
    a2 = pd.DataFrame(a.groupby('list')['points'].agg(['mean','count']))
    a2.sort_values(by = ['mean'],ascending = False,inplace = True)
    total= a2['count'].sum()
    a2['likelyhood'] = a2['count']/total
    #a2 likelyhood is the likelhood that you could draft that specific order 
    mean_likelyhood = a2['likelyhood'].mean()
    std_likelyhood = a2['likelyhood'].std()
    
    mean_2 = mean_likelyhood + std_likelyhood *2
    filt_a2 = a2[a2['likelyhood'] <mean_2]
    # fiter the data so that the Best possible score is that is likely to happen
    #within 2 std dev of the mean, meaning it has would be 
    best[item] = filt_a2['mean'].iloc[0:26].mean()
    std_best[item]  = filt_a2['mean'].iloc[0:26].std()
    
    mean_5 = mean_likelyhood + std_likelyhood * 0.5
    filt_a2_likely = a2[a2['likelyhood'] <mean_5]
    
    most_likely[item] = filt_a2_likely['mean'].iloc[0:26].mean()
    std_most_likely[item]  = filt_a2_likely['mean'].iloc[0:26].std()
    list_of_options = {}
    # plt.figure()
    # plt.hist(a2['likelyhood'])
    # plt.title(str(item) + " possition unfiltered")
    # plt.figure()
    # plt.hist(filt_a2['likelyhood'])
    # plt.title(str(item) + " possition filtered")
    #best[item] = a2['count'][0:26]  

    import numpy as np
    import statsmodels.api as sm
    import pylab as py
      
    # np.random generates different random numbers
    # whenever the code is executed
    # Note: When you execute the same code 
    # the graph look different than shown below.
      
    # Random data points generated
    #data_points = np.random.normal(0, 1, 100)    
      
    sm.qqplot(a2['mean'], line ='q')
    py.show()
    save_a2s[item] = a2['mean']

    
#     if item == Team_Stat:
#         output= pd.DataFrame(value,columns = ('list','points'))
#         team_strat = pd.DataFrame(output.groupby('list')['points'].agg(['mean','count']))
#         team_strat.sort_values(by = ['mean'],ascending = False,inplace = True)
#         team_strat.reset_index(inplace =True)
# plt.bar(team_strat['list'][0:3],team_strat['mean'][0:3])
# In[]:
import scipy.stats as stats
import pandas as pd
# perform two-sided test. You can use 'greater' or 'less' for one-sided test4
man_whitt =  {}

for i in range(1,11):
    lists = []

    for j in range(i+1,12):
        
        p_value = stats.mannwhitneyu(x=save_a2s[i], y=save_a2s[j], alternative = 'greater')[1]
           # if p_value< 0.05:
        print(i,j,p_value)
        




##What is the best strat this year 

# In[]:
all_selections =[]
for item in results:
    for j in results[item]:
        all_selections.append([j])
# In[]:

#stratagies
take_first={}
i = 0 
#item = team , value = 
for pair in all_selections:
    if len(take_first) ==0: 
        take_first[pair[0][0][0:3]] = [pair[0][1]]
        i = i +1
                             
    elif pair[0][0][0:3] in list(take_first.keys()):
        take_first[pair[0][0][0:3]].append(pair[0][1])
        i = i +1

    else:
        take_first[pair[0][0][0:3]] = [pair[0][1]]
        i = i +1

    
print(i/len(all_selections))

take_first_understand = []

for key in take_first:
    take_first_understand.append([key, np.percentile(take_first[key],75),round(len(take_first[key])/i*100,2)])
    
df = pd.DataFrame(take_first_understand)
plt_df = df[df[2]>1]
plt_df.sort_values(by= 2,inplace = True)
plt.figure()
plt.bar(plt_df[0],plt_df[1],color=['blue', 'blue','blue','blue', 'green', 'green', 'green'])    
plt.title('First Pick, average total')
# In[]:
i = 0 
take_second={}
#item = team , value = 
for pair in all_selections:
    if pair[0][0][0:3] in ['RB1','RB2','WR1']:
        
        if len(take_second) ==0: 
            take_second[pair[0][0][4:7]] = [pair[0][1]]
            i = i+1
        elif pair[0][0][4:7] in list(take_second.keys()):
            take_second[pair[0][0][4:7]].append(pair[0][1])
            i = i+1
        else:
            take_second[pair[0][0][4:7]] = [pair[0][1]]
            i = i+1
print(i/len(all_selections))

take_second_understand = []

for key in take_second:
    take_second_understand.append([key, np.percentile(take_second[key],75),round(len(take_second[key])/i*100,2)])

    
df = pd.DataFrame(take_second_understand)
plt_df = df[df[2]>1]
plt_df.sort_values(by= 2,inplace = True)
plt.figure()
plt.bar(plt_df[0],plt_df[1],color=['blue', 'blue','blue','blue', 'green', 'green', 'green'])        
plt.title('Second pick average totals')
        
      # In[]:
i = 0 
take_third={}
#item = team , value = 
for pair in all_selections:
    if pair[0][0][0:3] in ['RB1','RB2','WR1']:
        if pair[0][0][4:7] in ['WR2','RB3','RB4','QB1']:
            if len(take_third) ==0: 
                take_third[pair[0][0][8:11]] = [pair[0][1]]
                i = i+1
            elif pair[0][0][8:11] in list(take_third.keys()):
                take_third[pair[0][0][8:11]].append(pair[0][1])
                i = i+1
            else:
                take_third[pair[0][0][8:11]] = [pair[0][1]]
                i = i+1
print(i/len(all_selections))

take_third_understand = []

for key in take_third:
    take_third_understand.append([key, np.percentile(take_third[key],75),round(len(take_third[key])/i*100,2)])

    
df = pd.DataFrame(take_third_understand)
plt_df = df[df[2]>1]
plt_df.sort_values(by= 2,inplace = True)
plt.figure()
plt.bar(plt_df[0],plt_df[1],color=['blue', 'blue','blue','blue','blue', 'green', 'green', 'green'])        
plt.title('Third Pick average totals')
        
      # In[]:
i = 0 
take_four={}
#item = team , value = 
for pair in all_selections:
    if pair[0][0][0:3] in ['RB1','RB2','WR1']:
        if pair[0][0][4:7] in ['WR2','RB3','RB4','QB1']:
            if pair[0][0][8:11] in ['WR3','RB5','RB4','QB1']:
                if len(take_four) ==0: 
                    take_four[pair[0][0][12:15]] = [pair[0][1]]
                    i = i+1
                elif pair[0][0][12:15] in list(take_four.keys()):
                    take_four[pair[0][0][12:15]].append(pair[0][1])
                    i = i+1
                else:
                    take_four[pair[0][0][12:15]] = [pair[0][1]]
                    i = i+1
print(i/len(all_selections))

take_four_understand = []

for key in take_four:
    take_four_understand.append([key, np.percentile(take_four[key],75),round(len(take_four[key])/i*100,2)])

    
df = pd.DataFrame(take_four_understand)
plt_df = df[df[2]>1]
plt_df.sort_values(by= 2,inplace = True)
plt.figure()
plt.bar(plt_df[0],plt_df[1],color=['blue', 'blue','blue','blue','blue', 'green', 'green', 'green'])        
plt.title('Fourth pick average totals')
        

