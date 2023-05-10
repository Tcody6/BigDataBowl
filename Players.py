import pandas as pd
import numpy as np
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

pff = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\pffScoutingData.csv")
plays = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\plays.csv")
games = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\games.csv")
players = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\players.csv")
data = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\CoxInputs.csv")
data['forcedPlayEnd'] = np.where((data['Hit'] > 0.0) | (data['Hurry'] > 0.0) | (data['passResult'] == 'S'), 1, 0)

plays['ID_Full'] = plays['gameId'].astype(str) + "-" + plays['playId'].astype(str)

data = data.merge(plays[['ID_Full', "defensiveTeam"]]
                    , left_on='Play', right_on='ID_Full')

data.drop(['Play', 'Unnamed: 0', 'Hit', 'Hurry', 'passResult', 'gameId', 'playId'], axis=1, inplace=True)

dummies_OForm = pd.get_dummies(data["offenseFormation"], prefix = 'oForm')
data = pd.concat([data, dummies_OForm], axis = 1)
data = data.drop("offenseFormation", axis = 1)

dummies_dropBack = pd.get_dummies(data["dropBackType"], prefix = 'dropBack')
data = pd.concat([data, dummies_dropBack], axis = 1)
data = data.drop("dropBackType", axis = 1)

data = data.fillna(0)

T = data["timeTilRushEnd"]

E = data['forcedPlayEnd']

dataXOnly = data.drop(['forcedPlayEnd', 'timeTilRushEnd'], axis=1)

dataYOnly = np.zeros(8564, dtype={'names':('event', 'time'),
                          'formats':('?', 'f8')})

dataYOnly['event'] = E
dataYOnly['time'] = T

plays = plays.merge(games, on='gameId')

pff['ID_Full'] = pff['gameId'].astype(str) + "-" + pff['playId'].astype(str)
plays['ID_Full'] = plays['gameId'].astype(str) + "-" + plays['playId'].astype(str)

pff = pff.merge(plays, on="ID_Full")

rushers = pff[pff['pff_role'] == 'Pass Rush']['nflId'].value_counts().rename("Plays_Rushed")

rushers = rushers[rushers>29]

eligible_rush = rushers.index.values.tolist()

rush_merge = pff[pff['pff_role'] == 'Pass Rush'][['ID_Full', 'nflId', "defensiveTeam"]]

rush_merge = rush_merge[rush_merge['nflId'].isin(eligible_rush)]

count = 0

#for x in rushers:
    #did_rush = []
    #for y in plays.iterrows():
        #if len(pff[(pff['pff_role'] == 'Pass Rush') & (pff['nflId'] == x)]) > 0:
        #    did_rush.append(1)
        #else:
        #    did_rush.append(0)
    #plays[x] = did_rush
    #count += 1
    #print(count)


teams = plays["defensiveTeam"].unique()
average_predicted_sack_time = []
players_ordered = []
teams_ordered = []
count = 0

for x in teams:
    training = data[data["defensiveTeam"] == x].copy()
    team_filter = rush_merge[rush_merge['defensiveTeam'] == x]
    rushers = team_filter['nflId'].unique()
    for rusher in rushers:
        did_rush = []
        for play in training.iterrows():
            if len(pff[(pff['pff_role'] == 'Pass Rush') & (pff['nflId'] == rusher) & (pff['ID_Full'] == play[1]['ID_Full'])]) > 0:
                did_rush.append(1)
            else:
                did_rush.append(0)
        training[rusher] = did_rush
        count += 1
        print(count)
    T = training["timeTilRushEnd"]
    E = training['forcedPlayEnd']
    training = training.drop(['forcedPlayEnd', 'timeTilRushEnd', 'ID_Full', "defensiveTeam"], axis=1)
    dataYOnly = np.zeros(len(training), dtype={'names': ('event', 'time'),
                                      'formats': ('?', 'f8')})
    dataYOnly['event'] = E
    dataYOnly['time'] = T
    est_cph_tree = GradientBoostingSurvivalAnalysis(
        n_estimators=135, learning_rate=1, max_depth=2, random_state=0
    )
    est_cph_tree.fit(training, dataYOnly)

    players = np.eye(len(rushers))

    for y in players:
        predictedSackTimes = []
        for row in training.iterrows():
            feature_vector = np.array(row[1])
            feature_vector[23:len(players) + 24] = y
            sackTimes = est_cph_tree.predict_survival_function(feature_vector.reshape(1, -1))
            indexNeed = np.argmax(sackTimes[0].y < 0.5)
            predictedSackTimes.append(sackTimes[0].x[indexNeed])
        average_predicted_sack_time.append(sum(predictedSackTimes) / len(predictedSackTimes))
    for z in rushers:
        players_ordered.append(z)
        teams_ordered.append(x)
output_players = pd.DataFrame()

output_players['Player'] = players_ordered
output_players['Time'] = average_predicted_sack_time
output_players['Team'] = teams_ordered

output_players.to_csv("PlayersResults.csv")