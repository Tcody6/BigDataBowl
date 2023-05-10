import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
import warnings
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")

plays = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\plays.csv")
data = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\CoxInputs.csv")
data['forcedPlayEnd'] = np.where((data['Hit'] > 0.0) | (data['Hurry'] > 0.0) | (data['passResult'] == 'S'), 1, 0)

plays['ID_Full'] = plays['gameId'].astype(str) + "-" + plays['playId'].astype(str)

data = data.merge(plays[['ID_Full', 'possessionTeam']]
                    , left_on='Play', right_on='ID_Full')

test_play = data[data["ID_Full"] == '2021091203-3764']

data.drop(['Play', 'Unnamed: 0', 'Hit', 'Hurry', 'passResult', 'gameId', 'playId'], axis=1, inplace=True)

dummies_OForm = pd.get_dummies(data["offenseFormation"], prefix = 'oForm')
data = pd.concat([data, dummies_OForm], axis = 1)
data = data.drop("offenseFormation", axis = 1)

dummies_dropBack = pd.get_dummies(data["dropBackType"], prefix = 'dropBack')
data = pd.concat([data, dummies_dropBack], axis = 1)
data = data.drop("dropBackType", axis = 1)

dummies_OTeam = pd.get_dummies(data['possessionTeam'], prefix = 'OTeam')
data = pd.concat([data, dummies_OTeam], axis = 1)
data = data.drop('possessionTeam', axis = 1)

data = data.fillna(0)

T = data["timeTilRushEnd"]

E = data['forcedPlayEnd']

dataXOnly = data.drop(['forcedPlayEnd', 'timeTilRushEnd'], axis=1)

test_play = dataXOnly[dataXOnly["ID_Full"] == '2021091203-3764']
dataXOnly.drop(['ID_Full'], axis=1, inplace=True)
test_play.drop(['ID_Full'], axis=1, inplace=True)
dataYOnly = np.zeros(8564, dtype={'names':('event', 'time'),
                          'formats':('?', 'f8')})

dataYOnly['event'] = E
dataYOnly['time'] = T

X_train, X_test, y_train, y_test = train_test_split(dataXOnly, dataYOnly, test_size=0.25, random_state=0)

est_cph_tree = GradientBoostingSurvivalAnalysis(
    n_estimators=135, learning_rate=1, max_depth=2, random_state=0
)
est_cph_tree.fit(dataXOnly, dataYOnly)
cindex = est_cph_tree.score(dataXOnly, dataYOnly)
print(round(cindex, 3))

teams = np.eye(32)

count = 0
average_predicted_sack_time = []

for x in teams:
    predictedSackTimes = []
    for row in dataXOnly.iterrows():
        feature_vector = np.array(row[1])
        feature_vector[23:55] = x
        sackTimes = est_cph_tree.predict_survival_function(feature_vector.reshape(1, -1))
        indexNeed = np.argmax(sackTimes[0].y < 0.5)
        predictedSackTimes.append(sackTimes[0].x[indexNeed])
    average_predicted_sack_time.append(sum(predictedSackTimes) / len(predictedSackTimes))
    count +=1
    print(count)

play_time = est_cph_tree.predict_survival_function(np.array(test_play).reshape(1, -1))

output_offense = pd.DataFrame()

output_offense['Team'] = dummies_OTeam.columns
output_offense['Time'] = average_predicted_sack_time

output_offense.to_csv("OffenseTeams.csv")