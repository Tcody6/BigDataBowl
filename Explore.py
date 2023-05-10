import pandas as pd
import numpy as np

plays = pd.read_csv("C:\\Users\\19197\\PycharmProjects\\BigDataBowl\\plays.csv")
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
for row in plays[100:150].iterrows():
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

Times = pd.DataFrame(time_til_sack)

Times['Play'] = Times['gameId'].astype(str) + "-" + Times['playId'].astype(str)

Times.to_csv('PlayTimes.csv')

