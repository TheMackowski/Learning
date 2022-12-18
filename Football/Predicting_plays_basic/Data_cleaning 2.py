# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 07:50:02 2021

@author: MackowskiJ
"""
# In[]:
import numpy as np 

import pandas as pd
c2021 = pd.read_excel('Head Coaches_clean.xlsx','2021')
c2021['year'] = 2021
c2020 = pd.read_excel('Head Coaches_clean.xlsx','2021')
c2020['year'] = 2020
c2019 = pd.read_excel('Head Coaches_clean.xlsx','2019')
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
clean_2016['OffenseTeam'] = clean_2016['OffenseTeam'].str.replace('LV','OAK')
clean_2016['OffenseTeam'] = clean_2016['OffenseTeam'].str.replace('NO','NOR')
clean_2016['OffenseTeam'] = clean_2016['OffenseTeam'].str.replace('SF','SFO')
clean_2016['OffenseTeam'] = clean_2016['OffenseTeam'].str.replace('LARC','LAC')


clean_2016['DefenseTeam'] = clean_2016['DefenseTeam'].str.replace('TB','TAM')
clean_2016['DefenseTeam'] = clean_2016['DefenseTeam'].str.replace('NE','NWE')
clean_2016['DefenseTeam'] = clean_2016['DefenseTeam'].str.replace('GB','GNB')
clean_2016['DefenseTeam'] = clean_2016['DefenseTeam'].str.replace('KC','KAN')
clean_2016['DefenseTeam'] = clean_2016['DefenseTeam'].str.replace('LA','LAR')
clean_2016['DefenseTeam'] = clean_2016['DefenseTeam'].str.replace('LV','OAK')
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

data_with_labels = pd.merge(data,type_of_game, how = 'left',left_index=True, right_index=True)
colors = ['red','green','blue','black'] #,'purple','yellow']


# plt.scatter(data_with_labels['count_pass_offense'],data_with_labels['count_rush_offense'],c = data_with_labels.iloc[:,-1],cmap=mpl.colors.ListedColormap(colors))
# plt.figure()
# plt.scatter(data_with_labels['count_pass_offense'],data_with_labels['count_rush_d'],c = data_with_labels.iloc[:,-1],cmap=mpl.colors.ListedColormap(colors))

# plt.figure()
# plt.scatter(data_with_labels['count_rush_offense'],data_with_labels['count_rush_d'],c = data_with_labels.iloc[:,-1],cmap=mpl.colors.ListedColormap(colors))

# In[]:

import re
from datetime import datetime
game_makers = {}
for game in pbp_2016['GameId'].unique():
    temp_df = pbp_2016[pbp_2016['GameId'] ==game]
    players_in_game = []
    for row,stuff in temp_df.iterrows():
        
        apple = stuff['Description']
        try:
            match = re.search(r'-',apple)
            all_ =  re.findall(r'-[A-Za-z].[A-Za-z]*',apple)
            players_in_game.extend(all_)
        except:
            print('funny buisness at row ' +str(row))
    layer_strings = " ".join(players_in_game)
    game_makers[game] = layer_strings
    

# In[]:

best_defense_players = pd.read_excel('Defense_2019.xlsx')
best_defense_players.drop_duplicates(subset= ['Tm','Pos'] , keep = 'first',inplace = True)
# In[]:
from statistics import mean
avg_age_d ={}
avg_rank_d = {}
player_list_d = {}
Teams = ['ARI', 'HOU', 'GNB', 'CAR', 'CLE', 'SFO', 'NOR', 'SEA', 'IND',
       'DAL', 'TAM', 'DET', 'NYG', 'CIN', 'WAS', 'LAR', 'MIA', 'ATL',
       'TEN', 'PIT', 'CHI', 'BUF', 'MIN', 'LAC', 'DEN', 'OAK', 'PHI',
       'KAN', 'NYJ', 'NWE', 'JAX', 'BAL']
for team in Teams:

    temp = best_defense_players[best_defense_players['Tm']== team]
    list_of_players = temp['Last_Name']
    temp_list = list(list_of_players)
    
    
#    player_strings = " ".join(temp_list)
    player_list_d[team] = temp_list
    avg_age_d[team] = mean(temp['Age'])
    avg_rank_d[team] = mean(temp['Rk'])
    
# player_list_d['LVR'] = player_list_d.pop('OAK')
# In[]:

best_of_players = pd.read_excel('Offense_2019_2.xlsx')
best_of_players.drop_duplicates(subset= ['Tm','FantPos'] , keep = 'first',inplace = True)
best_of_players.dropna(how = 'any',axis = 0, subset =['FantPos'],inplace = True)
# In[]:
from statistics import mean
player_list_o = {}
avg_age_o ={}
avg_rank_o = {}
pos_list_o = {}
rnk_list_o = {}

Teams = ['ARI', 'HOU', 'GNB', 'CAR', 'CLE', 'SFO', 'NOR', 'SEA', 'IND',
       'DAL', 'TAM', 'DET', 'NYG', 'CIN', 'WAS', 'LAR', 'MIA', 'ATL',
       'TEN', 'PIT', 'CHI', 'BUF', 'MIN', 'LAC', 'DEN', 'OAK', 'PHI',
       'KAN', 'NYJ', 'NWE', 'JAX', 'BAL']
##Oakland to LVR

for team in Teams:
    temp = best_of_players[best_of_players['Tm']== team]
    list_of_players = temp['Last_Name']
    temp_list = list(list_of_players)
    temp_pos = list(temp['FantPos'])
    temp_ranks = list(temp['Rk'])
    
#    player_strings = " ".join(temp_list)
    player_list_o[team] = temp_list    
    avg_age_o[team] = mean(temp['Age'])
    avg_rank_o[team] = mean(temp['Rk'])
    pos_list_o[team] = temp_pos    
    rnk_list_o[team] = temp_ranks    
    
    
    
    
    
    
    
    
    
    
    
    
# In[]:
Games = df[['GameId','OffenseTeam','DefenseTeam_x']]
Games.drop_duplicates(keep = 'first',inplace = True)
Games['O_playing'] = 0.0000
Games['D_playing'] = 0.0
Games['rb_playing'] = 0
Games['qb_playing'] = 0
Games['wr_playing'] = 0
Games['rb_rank']= 600
Games['qb_rank']= 600
Games['wr_rank']= 600



# In[]:


##Did all the best players play in the game

#each game id_check all defensive players played that are on that team 
list_game_ids = Games['GameId'].drop_duplicates(keep = 'first')
for game in (list_game_ids):
    temp = Games[Games['GameId']== game]
    first_index =temp.index[0] 
    second_index = temp.index[1]
    subset = Games[Games['GameId']== game].reset_index()
    team_1_o = subset['OffenseTeam'][0]
    team_1_d = subset['DefenseTeam_x'][0] 
    team_2_o =subset['OffenseTeam'][1]
    team_2_d =subset['DefenseTeam_x'][1]
    

    o1_i = 0
    rb_t1 = 0
    wr_t1 = 0
    qb_t1 = 0
    te_t1 = 0
    rb_t1_rank = 0
    qb_t1_rank = 0
    wr_t1_rank = 0
    for item in player_list_o[team_1_o]:
        if item.lower() in  game_makers[game].lower():
            ind = player_list_o[team_1_o].index(item)
            player_pos = pos_list_o[team_1_o][ind]
           
            if player_pos =='RB':
                rb_t1 = 1
                rb_t1_rank = rnk_list_o[team_1_o][ind]
            elif player_pos =='WR':
                wr_t1 = 1
                wr_t1_rank = rnk_list_o[team_1_o][ind]
            elif player_pos =='QB':
                qb_t1 = 1
                qb_t1_rank = rnk_list_o[team_1_o][ind]

            elif player_pos =='TE':
                te_t1 = 1
            o1_i+=1
    len_t1o = len(player_list_o[team_1_o])
    
    d1_i = 0
    for item in player_list_d[team_1_d]:
        if item.lower() in  game_makers[game].lower():
            
            d1_i+=1
    len_t1d = len(player_list_d[team_1_d])
    o2_i = 0
    rb_t2 = 0
    wr_t2 = 0
    qb_t2 = 0
    te_t2 = 0
    rb_t2_rank = 0
    qb_t2_rank = 0
    qb_t2_rank  = 0 
    for item in player_list_o[team_2_o]:
        if item.lower() in  game_makers[game].lower():
            ind = player_list_o[team_2_o].index(item)
            player_pos = pos_list_o[team_2_o][ind]
           
            if player_pos =='RB':
                rb_t2 = 1
                rb_t2_rank = rnk_list_o[team_2_o][ind]
            elif player_pos =='WR':
                wr_t2 = 1
                wr_t2_rank = rnk_list_o[team_2_o][ind]
            elif player_pos =='QB':
                qb_t2 = 1
                qb_t2_rank = rnk_list_o[team_2_o][ind]
            elif player_pos =='TE':
                te_t2 = 1
            o2_i+=1
    len_t2o = len(player_list_o[team_2_o])        
    d2_i = 0
    for item in player_list_d[team_2_d]:
        if item.lower() in  game_makers[game].lower():
            d2_i+=1
    len_t2d = len(player_list_d[team_2_d])

    Games['O_playing'][first_index] = o1_i/len_t1o
    Games['D_playing'][first_index]= d1_i/len_t1d
    Games['O_playing'][second_index] = o2_i/len_t2o
    Games['D_playing'][second_index] = d2_i/len_t2d
    
    Games['rb_playing'][first_index] = rb_t1
    Games['qb_playing'][first_index] = qb_t1
    Games['wr_playing'][first_index] = wr_t1
    Games['rb_playing'][second_index] = rb_t2
    Games['wr_playing'][second_index] = wr_t2
    
    Games['rb_rank'][first_index] = rb_t1_rank
    Games['rb_rank'][second_index] = rb_t2_rank
    Games['qb_rank'][first_index] = qb_t1_rank
    Games['qb_rank'][second_index] = qb_t2_rank
    Games['wr_rank'][first_index] = wr_t1_rank
    Games['wr_rank'][second_index] = wr_t2_rank
# In[]:
Games['HOME_AWAY'] = df.sort_values(['GameId'], ascending=[True]) \
             .groupby(['GameId']) \
             .cumcount() + 1
# In[]:
weather = pd.read_excel('Weather.xlsx')
# w1 =weather[['Week','Game','Inside','Temp','Pricipitation','Wind Speed']]
# w2 =weather[['Week','Time (ET)','Inside','Temp','Pricipitation','Wind Speed']]
# w2.rename(columns = {'Time (ET)':'Game'})
# new_weather = pd.concat([w1,w2])1

#Games_weather = pd.merge(Games,weather, how = 'left', left_on = ['Offense])

# In[]:
game_to_week = pd.read_excel('dates_to_weekfootball.xlsx')

game_to_week['DAY'] = pd.to_datetime(game_to_week.iloc[:,-1], infer_datetime_format=True)
game_weeks = pd.merge(game_to_week,GameId_to_dates,how = 'right',on = 'DAY')

output_df = pd.merge(Games,game_weeks,how = 'left', on = 'GameId')
output_df.rename(columns={0:'week'},inplace = True)
# In[]:

output_df2 = pd.merge(output_df, weather, how = 'left', left_on = ['Week','OffenseTeam'], right_on = ['Week','Game'] )    

temp = pd.merge(output_df2,c2019[['Tm','Coach_Age']], how ='left',left_on = 'OffenseTeam', right_on = 'Tm')

temp.rename(columns = {'Coach_Age': 'oca'},inplace = True)
final_df = pd.merge(temp,c2019[['Tm','Coach_Age']], how ='left',left_on = 'DefenseTeam_x', right_on = 'Tm')
final_df.rename(columns = {'Coach_Age': 'dca'},inplace = True)

final_df = pd.merge(final_df,data_with_labels, how = 'left',left_index=True, right_index=True)

final_df.to_excel('final_clean_df_4_.xlsx')

