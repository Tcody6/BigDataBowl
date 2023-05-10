import pandas as pd
import numpy as np

pff = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\pffScoutingData.csv")
plays = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\plays.csv")
games = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\games.csv")

plays = plays.merge(games, on='gameId')

pff['ID_Full'] = pff['gameId'].astype(str) + "-" + pff['playId'].astype(str)
plays['ID_Full'] = plays['gameId'].astype(str) + "-" + plays['playId'].astype(str)

final = pd.DataFrame({'Play': pff.ID_Full.unique()})

rushers = pff[pff['pff_role'] == 'Pass Rush']['ID_Full'].value_counts().rename("Rushers")
blockers = pff[pff['pff_role'] == 'Pass Block']['ID_Full'].value_counts().rename("Blockers")
hit = pff[pff['pff_hit'] == 1]['ID_Full'].value_counts().rename("Hit")
hurry = pff[pff['pff_hurry'] == 1]['ID_Full'].value_counts().rename("Hurry")

final = final.join(rushers, "Play")
final = final.join(blockers, "Play")
final = final.join(hit, "Play")
final = final.join(hurry, "Play")
final['Rushers'] = final['Rushers'].fillna(0)
final['Rushers'] = final['Rushers'].astype(int)
final = final.merge(plays[['ID_Full', 'quarter', 'down', 'yardsToGo', 'gameClock', 'preSnapHomeScore',
                           'preSnapVisitorScore', 'offenseFormation', 'defendersInBox', 'dropBackType',
                           'pff_playAction', 'homeTeamAbbr', 'visitorTeamAbbr', 'possessionTeam', 'defensiveTeam', 'passResult']]
                    , left_on='Play', right_on='ID_Full')

final.drop(['ID_Full'], axis=1, inplace=True)

def convertTime(x):
    min, sec = map(int, x.split(':'))
    return min*60 + sec

final['TimeLeft'] = final['gameClock'].apply(convertTime)
final['TimeLeft'] = ((final['quarter']-4)*-1*15*60) + final['TimeLeft']

final.loc[final['TimeLeft'] < 0, 'TimeLeft'] = 0

final['OffenseScore'] = np.where(final['homeTeamAbbr'] == final['possessionTeam'], final['preSnapHomeScore'],
                                 final['preSnapVisitorScore'])

final['DefenseScore'] = np.where(final['homeTeamAbbr'] == final['defensiveTeam'], final['preSnapHomeScore'],
                                 final['preSnapVisitorScore'])

final['Score'] = final['OffenseScore'] - final['DefenseScore']

final.drop(['quarter', 'gameClock', 'preSnapHomeScore', 'preSnapVisitorScore', 'homeTeamAbbr', 'visitorTeamAbbr',
            'possessionTeam', 'defensiveTeam', 'OffenseScore', 'DefenseScore'], axis=1, inplace=True)

week1 = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\week1.csv")
week2 = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\week2.csv")
week3 = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\week3.csv")
week4 = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\week4.csv")
week5 = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\week5.csv")
week6 = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\week6.csv")
week7 = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\week7.csv")
week8 = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\week8.csv")

Tracking = pd.concat([week1, week2, week3, week4, week5, week6, week7, week8])

time_til_sack = []
count = 0

for row in plays.iterrows():
    try:
        snap = Tracking.loc[(Tracking['gameId'] == row[1]['gameId']) & (Tracking['playId'] == row[1]['playId'])
                            & ((Tracking['event'] == 'ball_snap') | (Tracking['event'] == 'autoevent_ball_snap'))].iloc[0]['frameId']
    except:
        snap = 1
    try:
        end = Tracking.loc[(Tracking['gameId'] == row[1]['gameId']) & (Tracking['playId'] == row[1]['playId']) &
             ((Tracking['event'] == 'qb_sack') | (Tracking['event'] == 'qb_strip_sack') |
             (Tracking['event'] == 'run') | (Tracking['event'] == 'autoevent_passinterrupted') |
             (Tracking['event'] == 'pass_tipped') | (Tracking['event'] == 'autoevent_passforward') |
             (Tracking['event'] == 'pass_forward'))].iloc[0]['frameId']
    except:
        end = "NA"
    time = (end - snap) / 10
    time_til_sack.append([row[1]['gameId'], row[1]['playId'], time])
    count +=1
    print(count / 8558)

sackTimes = pd.DataFrame(time_til_sack, columns=['gameId', 'playId', 'timeTilRushEnd'])

sackTimes['Play'] = sackTimes['gameId'].astype(str) + "-" + sackTimes['playId'].astype(str)

final = final.merge(sackTimes, on='Play')

final.to_csv('CoxInputs.csv')