# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 08:08:24 2021

@author: 15869
"""

# In[]:
import numpy as np 

import pandas as pd
c2021 = pd.read_excel('Head Coaches_clean.xlsx','2021')
c2021['year'] = 2021
c2020 = pd.read_excel('Head Coaches_clean.xlsx','2021')
c2020['year'] = 2020
c2019 = pd.read_excel('Head Coaches_clean.xlsx','2021')
c2019['year'] = 2019

c2018 = pd.read_excel('Head Coaches_clean.xlsx','2021')
c2018['year'] = 2018

c2017 = pd.read_excel('Head Coaches_clean.xlsx','2021')
c2017['year'] = 2017

c2016 = pd.read_excel('Head Coaches_clean.xlsx','2021')
c2016['year'] = 2016
coaches = pd.concat([c2020,c2021,c2019,c2018,c2017,c2016])


c2019_old = c2019[c2019['career games']>=200]
c2019_old['Coach_Age'] = 3
c2019_new= c2019[c2019['career games']<=36]
c2019_new['Coach_Age'] = 1

c2019_middle = c2019[((c2019['career games']<200)  & (c2019['career games']>36)) ]

c2019_middle['Coach_Age'] = 2

c2019 = pd.concat([c2019_new,c2019_old,c2019_middle])

# In[]:
pbp_2016 = pd.read_csv('pbp-2019.csv')
pbp_2016.dropna(axis = 1, how = 'all',inplace = True)
pbp_2016.dropna(axis = 0, how = 'any',subset = ['OffenseTeam'],inplace = True)

summations_games2016 = pbp_2016[['GameId', 'OffenseTeam',
       'DefenseTeam',  'Yards',
       'PlayType', 'IsRush', 'IsPass', 'IsIncomplete' ]].groupby(['GameId','OffenseTeam','DefenseTeam','PlayType']).sum().reset_index()

summations_games2016.columns = list(map(''.join, summations_games2016.columns.values))
clean_2016 = summations_games2016[summations_games2016['Yards'] != 0 ]
GameId_to_dates = pbp_2016[['GameId','GameDate']]
GameId_to_dates.drop_duplicates(subset = 'GameId',inplace = True)
GameId_to_dates['DAY'] = pd.to_datetime(GameId_to_dates['GameDate'], infer_datetime_format=True)
# In[]:
clean_2016['yards_pass_offense'] = clean_2016['Yards']* ((clean_2016['IsPass']>1) * 1)
clean_2016['yards_rush_offense'] = clean_2016['Yards']* ((clean_2016['IsRush']>1) * 1)
clean_2016['count_pass_offense'] = clean_2016['IsPass'].copy()
clean_2016['count_rush_offense'] = clean_2016['IsRush'].copy()
# In[]:
clean_2016['OffenseTeam'] = clean_2016['OffenseTeam'].str.replace('TB','TAM')
clean_2016['OffenseTeam'] = clean_2016['OffenseTeam'].str.replace('NE','NWE')
clean_2016['OffenseTeam'] = clean_2016['OffenseTeam'].str.replace('GB','GNB')
clean_2016['OffenseTeam'] = clean_2016['OffenseTeam'].str.replace('KC','KAN')
clean_2016['OffenseTeam'] = clean_2016['OffenseTeam'].str.replace('LA','LAR')
clean_2016['OffenseTeam'] = clean_2016['OffenseTeam'].str.replace('LV','LVR')
clean_2016['OffenseTeam'] = clean_2016['OffenseTeam'].str.replace('NO','NOR')
clean_2016['OffenseTeam'] = clean_2016['OffenseTeam'].str.replace('SF','SFO')
clean_2016['OffenseTeam'] = clean_2016['OffenseTeam'].str.replace('LARC','LAC')


clean_2016['DefenseTeam'] = clean_2016['DefenseTeam'].str.replace('TB','TAM')
clean_2016['DefenseTeam'] = clean_2016['DefenseTeam'].str.replace('NE','NWE')
clean_2016['DefenseTeam'] = clean_2016['DefenseTeam'].str.replace('GB','GNB')
clean_2016['DefenseTeam'] = clean_2016['DefenseTeam'].str.replace('KC','KAN')
clean_2016['DefenseTeam'] = clean_2016['DefenseTeam'].str.replace('LA','LAR')
clean_2016['DefenseTeam'] = clean_2016['DefenseTeam'].str.replace('LV','LVR')
clean_2016['DefenseTeam'] = clean_2016['DefenseTeam'].str.replace('NO','NOR')
clean_2016['DefenseTeam'] = clean_2016['DefenseTeam'].str.replace('SF','SFO')
clean_2016['DefenseTeam'] = clean_2016['DefenseTeam'].str.replace('LARC','LAC')
#
# In[]:


df_off = clean_2016[['GameId', 'OffenseTeam',
       'DefenseTeam','yards_pass_offense','yards_rush_offense',
       'count_pass_offense','count_rush_offense']].groupby(['GameId', 'OffenseTeam',
       'DefenseTeam']).sum().reset_index()

df_off.columns = list(map(''.join, df_off.columns.values))

d_col_list = {'yards_pass_offense':'yards_pass_d',
       'yards_rush_offense':'yards_rush_d', 'count_pass_offense':'count_pass_d', 'count_rush_offense':'count_rush_d'}

defence_df  = df_off[['GameId','DefenseTeam', 'yards_pass_offense',
       'yards_rush_offense', 'count_pass_offense', 'count_rush_offense']].rename(columns = d_col_list)

df = pd.merge(df_off,defence_df,how = 'left', left_on = ['GameId','OffenseTeam'], right_on = ['GameId','DefenseTeam'])
# In[]:

data = df[['count_pass_offense','count_rush_offense','count_pass_d','count_rush_d']]
import matplotlib.pyplot as plt
import matplotlib as mpl




from sklearn.cluster import KMeans
import numpy as np

kmeans = KMeans(n_clusters=4, random_state=0).fit(data)
type_of_game = pd.DataFrame(kmeans.labels_) 

df = pd.merge(data,type_of_game, how = 'left',left_index=True, right_index=True)

# In[]:
list_of_std = []
list_of_counts = []
for i in range(2,15,1):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
    type_of_game = pd.DataFrame(kmeans.labels_) 

    df = pd.merge(data,type_of_game, how = 'left',left_index=True, right_index=True)
    list_of_std.append(df.groupby(0).std().T.mean().mean())
    list_of_counts.append(df.groupby(0).count().min().min())
fig,ax = plt.subplots()

ax.plot(range(2,15,1),list_of_std,'green',marker = 'o')
ax2=ax.twinx()
ax2.plot(range(2,15,1),list_of_counts,'orange',marker = 'o')
plt.title('Amount of Groups counts and Standard Deviations')
plt.xlabel('Number of Groups')
ax.set_ylabel('Average Standard Deviations',color = 'orange')
ax2.set_ylabel('Minimum number of games in each grouping',color= 'green')
plt.axvline(10)

# In[]:
    
kmeans = KMeans(n_clusters=4, random_state=0).fit(data)
type_of_game = pd.DataFrame(kmeans.labels_) 

df = pd.merge(data,type_of_game, how = 'left',left_index=True, right_index=True)




colors = ['red','green','blue','black','purple','yellow','darksalmon','greenyellow']#,'royalblue','hotpink','brown','gold']


plt.scatter(df['count_pass_offense'],df['count_rush_offense'],c = df.iloc[:,-1],cmap=mpl.colors.ListedColormap(colors))
plt.figure()
plt.scatter(df['count_pass_offense'],df['count_pass_d'],c = df.iloc[:,-1],cmap=mpl.colors.ListedColormap(colors))

plt.figure()
plt.scatter(df['count_rush_offense'],df['count_rush_d'],c = df.iloc[:,-1],cmap=mpl.colors.ListedColormap(colors))

# In[]:
fig, axs = plt.subplots(4, 8,sharex='col',sharey='row')

for tog in range(8):
    
    hist_temp = df[df[0] == tog]
    axs[0,tog].hist(hist_temp['count_rush_offense'])
    axs[1,tog].hist(hist_temp['count_pass_offense'])
    axs[2,tog].hist(hist_temp['count_rush_d'])
    axs[3,tog].hist(hist_temp['count_pass_d'])
    
axs[0,0].set_ylabel('Rushes O')
axs[1,0].set_ylabel('Passes O')
axs[2,0].set_ylabel('Rushes D')
axs[3,0].set_ylabel('Passes D')
axs[0,3].set_title('Game')
axs[0,4].set_title('Types')



   






