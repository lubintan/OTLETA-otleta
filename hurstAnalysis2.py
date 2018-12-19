from math import pi
from datetime import datetime
from datetime import timedelta
import pandas_datareader as web
import pandas as pd
import numpy as np
import plotly
from plotly.graph_objs import Layout, Scatter, Line, Ohlc, Figure, Histogram, Bar, Table
import plotly.figure_factory as ff
import scipy as sp
import time
from trendFunctions import *

GAUGE = []


def removeDuplicates(cycles):
    cycles = set(cycles)
    cycles = list(cycles)
    cycles.sort()
    return cycles


def checkNoDuplicates(pointsList):
    if len(pointsList) == len(set(pointsList)):
        return True
    return False


def fillUpTopLevel(prevCycle, topCycle, compareList, prevPeriod, ma, bufferFactor=0.3):
    pointScore = []
    pointDict = {}

    # print(prevCycle)

    for i in range(len(prevCycle)):
        temp = []
        for eachPointList in compareList:
            temp.append(getScore(prevCycle[i], eachPointList, bufferFactor * prevPeriod, ma))
        thisScore = np.sum(temp)
        pointScore.append(thisScore)
        pointDict[thisScore] = i

    strongestPointIdx = pointDict[max(pointScore)]

    # print(pointDict)
    # exit()

    # ratio 3:1
    # fill by 3rds to the right
    i = strongestPointIdx
    while i < len(prevCycle):
        topCycle.append(prevCycle[i])
        i += 3

    # fill by 3rds to the left
    i = strongestPointIdx - 3
    while i >= 0:
        topCycle.append(prevCycle[i])
        i -= 3

    topCycle.sort()
    return topCycle


def getScore(point, otherPoints, buffer, ma):
    left = point - (buffer / 2)
    right = point + (buffer / 2)
    scoreList = []

    for idx, row in otherPoints.iterrows():
        if (left < idx) and (idx < right):
            # horizontal diff smaller better.
            # vertical: the lower the point, the better.
            # score: the higher the better
            horDiff = abs(point - idx)
            vertical = ma.iloc[idx].movingAverage - row.point
            score = vertical - horDiff
            scoreList.append(score)

    return np.sum(scoreList)


def getAvgPeriods(cyclePoints, left, right):
    if len(cyclePoints) < 2: return 'Not Applicable'

    cyclePoints = set(cyclePoints)
    cyclePoints = list(cyclePoints)
    cyclePoints.sort()

    periods = []
    for i in range(len(cyclePoints) - 1):
        if (left <= cyclePoints[i]) and (cyclePoints[i + 1] <= right):
            periods.append(cyclePoints[i + 1] - cyclePoints[i])

    return '%.2f' % (np.mean(periods)), np.mean(periods)


def postDownFiller(longerCycle, shorterCycle, dfLowLp, dfLowLp1Deg, fillDown=[]):
    for i in range(len(longerCycle) - 1):
        if not checkPointsInPeriod(longerCycle[i], longerCycle[i + 1], pointList=shorterCycle):
            newPoint = addPointBetween(longerCycle[i], longerCycle[i + 1], dfLowLp, dfLowLp1Deg)
            if newPoint != longerCycle[i]:
                shorterCycle.append(newPoint)
                # fill down
                for eachCycle in fillDown:
                    eachCycle.append(newPoint)
                    eachCycle.sort()
    shorterCycle.sort()


def addPointBetween(left, right, dfLowLp, dfLowLp1Deg):
    tempY = []
    tempDict = {}

    mid = (left + right) / 2.0

    for idx, row in dfLowLp.iterrows():
        if (left < idx) and (idx < right):
            dist = abs(idx - mid)
            tempY.append(dist)
            tempDict[dist] = idx

    if len(tempY) > 0:
        return tempDict[min(tempY)]

    for idx, row in dfLowLp1Deg.iterrows():
        if (left < idx) and (idx < right):
            dist = abs(idx - mid)
            tempY.append(dist)
            tempDict[dist] = idx

    if len(tempY) > 0:
        return tempDict[min(tempY)]

    return int((left + right) / 2)


def fillUp(longerPeriodCycle, shorterPeriodCycle, nominal, lastPoint, timeTolerance=0.4):
    longerPeriodCycle.sort()
    shorterPeriodCycle.sort()
    rightDist = (1 + timeTolerance) * nominal

    # print('dist:', rightDist)

    # fill right
    for eachLow in longerPeriodCycle:
        if (eachLow + rightDist) > lastPoint: break
        if not checkPointsInPeriod(eachLow, (eachLow + rightDist), longerPeriodCycle):
            newPoint = addPointFromShorterCycle(eachLow + nominal, shorterPeriodCycle, longerPeriodCycle)
            if newPoint != eachLow: longerPeriodCycle.append(newPoint)
        longerPeriodCycle.sort()
        # print(longerPeriodCycle)
        # print()

    # fill left
    leftLows = [longerPeriodCycle[0]]
    for eachLow in leftLows:
        if eachLow < rightDist: break
        if not checkPointsInPeriod(eachLow - rightDist, eachLow, leftLows, ):
            newPoint = addPointFromShorterCycle(eachLow - nominal, shorterPeriodCycle, longerPeriodCycle)
            if newPoint != eachLow: leftLows.append(newPoint)
        leftLows.sort()
        leftLows.reverse()

    for each in leftLows:
        longerPeriodCycle.append(each)

    longerPeriodCycle = set(longerPeriodCycle)
    longerPeriodCycle = list(longerPeriodCycle)
    longerPeriodCycle.sort()
    shorterPeriodCycle.sort()

    # print(longerPeriodCycle)


def addPointFromShorterCycle(point, shorterCycle, longerCycle):
    distDict = {}
    distFromPoint = []

    # print('shorter:',shorterCycle)

    for i in range(len(shorterCycle)):
        dist = abs(shorterCycle[i] - point)
        if shorterCycle[i] in longerCycle: continue

        distFromPoint.append(dist)
        distDict[dist] = shorterCycle[i]

    value = distDict[min(distFromPoint)]
    # print('value:',value, 'point:', point)
    return value


def gapfillers(cycle, dfLowLp1Deg, dfLowLp, maLp, lastPoint, timeTolerance=0.4, nominal=10):
    cycle.sort()
    rightDist = (1 + timeTolerance) * nominal
    # print('distance:', rightDist)

    # fill to the right
    for eachLow in cycle:
        if (eachLow + rightDist) > lastPoint: break
        # print(cycle)
        # time.sleep(.5)
        if not checkPointsInPeriod(eachLow, (eachLow + rightDist), cycle, ):
            cycle.append(addPoint(eachLow, (eachLow + rightDist), maLp, dfLowLp1Deg, dfLowLp, nominal))
        cycle.sort()

    # fill to the left
    leftLows = [cycle[0]]
    for eachLow in leftLows:
        if eachLow < rightDist: break
        # print('*******')
        # print(leftLows)
        if not checkPointsInPeriod(eachLow - rightDist, eachLow, leftLows, ):
            leftLows.append(addPoint((eachLow - rightDist), eachLow, maLp, dfLowLp1Deg, dfLowLp, nominal))
        leftLows.sort()
        leftLows.reverse()

    for each in leftLows:
        cycle.append(each)

    cycle.sort()


def addPoint(left, right, smallMaLp, dfLowLp1Deg, dfLowLp, nominal=10):
    halfWin = int(nominal / 3)
    tempDict = {}
    tempY = []
    # print()
    # print('left:',left,'right:',right)

    # try to find lows close to the MA low. if don't have, find lows from deg2 lows. then deg 1 lows.
    for maIdx, maRow in smallMaLp.iterrows():

        if (left < maIdx) and (maIdx < right):
            nextLeft = maIdx - halfWin
            nextRight = maIdx + halfWin

            for index, row in dfLowLp.iterrows():
                if (nextLeft < index) and (index < nextRight) and (index != left) and (index != right):
                    tempY.append(row.point)
                    tempDict[row.point] = index
    if len(tempY) > 0:
        # print('1:', tempDict)
        return tempDict[min(tempY)]

    for index, row in dfLowLp.iterrows():
        if (left < index) and (index < right):
            tempY.append(row.point)
            tempDict[row.point] = index

    if len(tempY) > 0:
        # print('2:', tempDict)
        return tempDict[min(tempY)]

    for index, row in dfLowLp1Deg.iterrows():
        if (left < index) and (index < right):
            tempY.append(row.point)
            tempDict[row.point] = index

    if len(tempY) > 0:
        # print('3:', tempDict)
        return tempDict[min(tempY)]

    # print('4:', tempDict)
    return int((left + right) / 2)


def checkPointsInPeriod(left, right, pointList, ):
    for each in pointList:
        if (left < each) and (each < right):
            return True
    return False


def filterRatios(M54, M18, M9, W20, W10, lows):
    M54.sort()
    M18.sort()
    M9.sort()
    W20.sort()
    W10.sort()

    # removeExtras(3,M54,M18,lows)
    removeExtras(2, M18, M9, lows)
    removeExtras(2, M9, W20, lows)
    removeExtras(2, W20, W10, lows)


def removeExtras(numRatio, higherPeriodCycle, lowerPeriodCycle, lows):
    # print(higherPeriodCycle)
    # print(lowerPeriodCycle)
    # print()
    # print()

    for i in range(len(higherPeriodCycle)):
        # print(higherPeriodCycle[i], '**********')
        # print()
        if (i + 2) > len(higherPeriodCycle):
            break
        left = higherPeriodCycle[i]
        right = higherPeriodCycle[i + 1]
        # print(left,right)
        count = 0
        temp = []
        for each in lowerPeriodCycle:
            if (left < each) and (each < right):
                count += 1
                temp.append(each)
        # print(temp)
        # print()

        if count > (numRatio - 1):
            lowsToRemove = []
            lowDict = {}
            for each in temp:
                # print('each:',each)
                lowDict[lows.iloc[each]] = each
                lowsToRemove.append(lows.iloc[each])

            lowsToRemove.sort()
            # print(lowsToRemove)
            # print(lowDict)

            # sort ascending. Remove the lowest lows.
            for i in range(numRatio - 1):
                lowsToRemove.pop(0)

            for each in lowsToRemove:
                lowerPeriodCycle.remove(lowDict[each])


def checkLeftRight(barNum, perhaps, cyclesM, M_weeks, numCycles, cycleAllowance, totalDataLength, filldownList,
                   upperCycles=[], upperCyclePeriods=[]):
    # percentage of period that needs to be cleared for upper cycle before putting in new one
    allowPercent = 1 - cycleAllowance

    # check right side

    if (barNum + (M_weeks * numCycles)) < totalDataLength:
        lowerbound = barNum + (M_weeks * numCycles) - (M_weeks * cycleAllowance)
        upperbound = barNum + (M_weeks * numCycles) + (M_weeks * cycleAllowance)

        # print()

        perhaps2 = []
        lpDiffList = []
        for i in range(len(perhaps)):
            barData = perhaps[i]

            # print(barData[0], upperbound,lowerbound)

            if (barData[0] < upperbound) and (barData[0] > lowerbound):
                perhaps2.append(barData)
                lpDiffList.append(barData[1])
                # print('appended:',barData)

        if len(perhaps2) > 1:
            lowestDiff = min(lpDiffList)

            for each in perhaps2:
                if each[1] == lowestDiff:

                    # region: left filtering for 2:1, 3:1 ratio to the end
                    # #check that there's only 1 of this between 2 longer period cycle points
                    # count = 0
                    # higherPeriodRight = barNum + higherPeriodCycle
                    #
                    # for eachPoint in cyclesM:
                    #     if (eachPoint>barNum) and (eachPoint<higherPeriodRight):
                    #         count += 1
                    #
                    # if count ==0:
                    # endregion

                    # check if too close to upper cycles
                    pointsTooClose = False
                    for upperIdx in range(len(upperCycles)):
                        for eachUpperPoint in upperCycles[upperIdx]:
                            if abs(eachUpperPoint - each[0]) < (allowPercent * upperCyclePeriods[upperIdx]):
                                pointsTooClose = True

                    if pointsTooClose:
                        # remove from perhaps
                        for eachValue in perhaps:
                            if eachValue[0] == each[0]:
                                perhaps.remove(eachValue)
                        continue

                    # add point
                    cyclesM.append(each[0])
                    # fill down
                    value = cyclesM[-1]
                    for eachCycle in filldownList:
                        eachCycle.append(value)

                    # remove from perhaps
                    for eachValue in perhaps:
                        if eachValue[0] == value:
                            perhaps.remove(eachValue)


        elif len(perhaps2) == 1:

            # check if too close to upper cycles
            pointsTooClose = False
            for upperIdx in range(len(upperCycles)):
                for eachUpperPoint in upperCycles[upperIdx]:
                    if abs(eachUpperPoint - perhaps2[-1][0]) < (allowPercent * upperCyclePeriods[upperIdx]):
                        pointsTooClose = True

            if not pointsTooClose:
                cyclesM.append(perhaps2[-1][0])
                # fill down
                value = cyclesM[-1]
                for eachCycle in filldownList:
                    eachCycle.append(value)

            # remove from perhaps
            for eachValue in perhaps:
                if eachValue[0] == perhaps2[-1][0]:
                    perhaps.remove(eachValue)

    if (barNum > (numCycles * M_weeks)):
        lowerbound = barNum - (M_weeks * numCycles) - (M_weeks * cycleAllowance)
        upperbound = barNum - (M_weeks * numCycles) + (M_weeks * cycleAllowance)

        perhaps2 = []
        lpDiffList = []
        for i in range(len(perhaps)):
            barData = perhaps[i]
            if (barData[0] < upperbound) and (barData[0] > lowerbound):
                perhaps2.append(barData)
                lpDiffList.append(barData[1])

        if len(perhaps2) > 1:
            lowestDiff = min(lpDiffList)

            for each in perhaps2:
                if each[1] == lowestDiff:

                    # check if too close to upper cycles
                    pointsTooClose = False
                    for upperIdx in range(len(upperCycles)):
                        for eachUpperPoint in upperCycles[upperIdx]:
                            if abs(eachUpperPoint - each[0]) < (allowPercent * upperCyclePeriods[upperIdx]):
                                pointsTooClose = True

                    if pointsTooClose:
                        # remove from perhaps
                        for eachValue in perhaps:
                            if eachValue[0] == each[0]:
                                perhaps.remove(eachValue)
                        continue

                    # add point
                    cyclesM.append(each[0])
                    # fill down
                    value = cyclesM[-1]
                    for eachCycle in filldownList:
                        eachCycle.append(value)

                    # remove from perhaps
                    for eachValue in perhaps:
                        if eachValue[0] == value:
                            perhaps.remove(eachValue)


        elif len(perhaps2) == 1:

            # check if too close to upper cycles
            pointsTooClose = False
            for upperIdx in range(len(upperCycles)):
                for eachUpperPoint in upperCycles[upperIdx]:
                    if abs(eachUpperPoint - perhaps2[-1][0]) < (allowPercent * upperCyclePeriods[upperIdx]):
                        pointsTooClose = True

            if not pointsTooClose:
                cyclesM.append(perhaps2[-1][0])
                # fill down
                value = cyclesM[-1]
                for eachCycle in filldownList:
                    eachCycle.append(value)

            # remove from perhaps
            for eachValue in perhaps:
                if eachValue[0] == perhaps2[-1][0]:
                    perhaps.remove(eachValue)

    return perhaps


def getPotentials(maLp, dfLowLp, period, upperband, lowerband, bufferFactor=0.2, envFactor=0.1):
    # maLp = maLp.copy()
    # maLp = pd.DataFrame(maLp)
    # maLp.columns = ['point']

    buffer = bufferFactor * period
    envWidth = upperband.dropna().iloc[0] - lowerband.dropna().iloc[0]
    lpDiffList = []
    perhaps = []

    for i1, r1 in maLp.iterrows():
        for i2, r2 in dfLowLp.iterrows():
            if (abs(i1 - i2) < buffer) and (r2.point < (r1.point + (envFactor * envWidth))):
                perhaps.append([i2, (r2.point - r1.point)])
                lpDiffList.append(r2.point - r1.point)
                break

    return perhaps, lpDiffList


def inverseMa(ma, data):
    inverse = data - ma

    return Bar(name='inverse MA', x=inverse.index, y=inverse, )


def highestPointFinder1Deg(y):
    hpMask = (
            (y.shift(-1) <= y)
            & (y.shift(1) <= y)
    )

    return y[hpMask]


def lowestPointFinder1Deg(y):
    lpMask = (
            (y.shift(-1) >= y)
            # & (y.shift(-2) > y.shift(-1))
            & (y.shift(1) >= y)
        # & (y.shift(2) >= y.shift(1))
    )

    return y[lpMask]


def lowestPointFinder(y):
    gradients = y - y.shift(1)

    # lpMask = (
    #         ((gradients.shift(-1)>0)
    #          # & (gradients.shift(-2)>0)
    #          )
    #           &
    #          ((gradients<0)
    #           # & gradients.shift(1)<(0)
    #           )
    #          ) | (gradients==0)

    lpMask = (
            (y.shift(-1) >= y)
            & (y.shift(-2) > y.shift(-1))
            & (y.shift(1) >= y)
            & (y.shift(2) >= y.shift(1))
    )

    return y[lpMask]


def envelopeFinder(df, ma, allowMargin=0.1):
    diff = max(df.High - df.Low) / 2
    step = 0.05
    adjust = 0.5
    maSize = len(ma.dropna())
    # allowMargin = 0.1
    allowNumber = allowMargin * maSize

    upperband = ma.movingAverage + (diff * adjust)
    lowerband = ma.movingAverage - (diff * adjust)

    while True:
        pokingHighs = (df.High > upperband).sum()
        pokingLows = (df.Low < lowerband).sum()

        if (pokingHighs < allowNumber) and (pokingLows < allowNumber):
            break
        # widen the bands
        adjust += step
        upperband = ma.movingAverage + (diff * adjust)
        lowerband = ma.movingAverage - (diff * adjust)

    return upperband, lowerband


def movingAverage(data, window, win_type=None, dates=None):
    # print(data)
    ma = data.rolling(window=window, center=True, win_type=win_type).mean()
    # print(window, '**********')
    # print(ma)
    # print()

    ma = pd.DataFrame(ma)
    ma.columns = ['movingAverage']
    ma['original'] = data
    # ma['date'] = dates
    # print(ma)
    # exit()
    # ma = ma.dropna()

    # print(ma)
    # exit()

    return ma


def extendMA(ma, period):
    noNA = ma.dropna()
    point2 = noNA.iloc[-1].movingAverage
    point1 = noNA.iloc[-2].movingAverage

    index2 = noNA.index[-1]
    index1 = noNA.index[-2]

    # y = mx+c
    gradient = point2 - point1  # because the x2-x1 here is 1.
    # c = y-mx
    yIntercept = point1 - (index1 * gradient)

    halfPeriod = int(period / 2)
    # print(halfPeriod)
    # exit()
    for index, row in ma.iloc[-1 * halfPeriod::].iterrows():
        row.movingAverage = gradient * index + yIntercept

    # print(ma.tail())
    return ma


def plotVerticalSubs(filename, mainTrace, title, lastIdx, avgList=[], others=[], anchorPoint=None):
    numRows = len(others)
    fig = plotly.tools.make_subplots(
        rows=numRows + 1, cols=1, shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.001,
        row_width=[0.1] * numRows + [0.8])

    for eachTrace in mainTrace:
        fig.append_trace(eachTrace, 1, 1)

    fig.append_trace(anchorPoint, 4, 1)
    for i in range(numRows):
        fig.append_trace(others[i], i + 2, 1)

    fig['layout'].update(
        xaxis=dict(rangeslider=dict(visible=False), gridwidth=2, mirror='ticks', nticks=30, showgrid=True),
        yaxis2=dict(showticklabels=False, title='Inverse MA'),
        yaxis3=dict(showticklabels=False, title='240W'),
        yaxis4=dict(showticklabels=False, title='80W'),
        yaxis5=dict(showticklabels=False, title='40W'),
        yaxis6=dict(showticklabels=False, title='20W'),
        yaxis7=dict(showticklabels=False, title='10W'),
        title=title,
        # annotations=[dict(
        #                    x=lastIdx,
        #                    y=0,
        #                    xref='x3',
        #                    yref='y3',
        #                    text='Avg: '+ avgList[0] + 'weeks',
        #                    showarrow=False,
        #                    align='right',
        #                ),
        #                 dict(
        #                     x=lastIdx,
        #                     y=0,
        #                     xref='x4',
        #                     yref='y4',
        #                     text='Avg: '+ avgList[1] + 'weeks',
        #                     showarrow=False,
        #                     align='right',
        #                 ),
        #                 dict(
        #                     x=lastIdx,
        #                     y=0,
        #                     xref='x5',
        #                     yref='y5',
        #                     text='Avg: '+ avgList[2] + 'weeks',
        #                     showarrow=False,
        #                     align='right',
        #                 ),
        #                 dict(
        #                     x=lastIdx,
        #                     y=0,
        #                     xref='x6',
        #                     yref='y6',
        #                     text='Avg: '+ avgList[3] + 'weeks',
        #                     showarrow=False,
        #                     align='right',
        #                 ),
        #                 dict(
        #                     x=lastIdx,
        #                     y=0,
        #                     xref='x7',
        #                     yref='y7',
        #                     text='Avg: '+ avgList[4] + 'weeks',
        #                     showarrow=False,
        #                     align='right',
        #                 ),]
        )

    plotly.offline.plot(fig, output_type='file', filename=filename)


def main(tickerName):
    global GAUGE

    window = 5
    if tickerName == 'EURUSD':
        df = pd.read_excel('EURUSD Weekly Data for Swing Indicator.xlsx')[100:450]
        df.columns = ['Date', 'Close', 'Open', 'High', 'Low']

    elif tickerName == 'BA':
        df = pd.read_csv('BA.csv')

    else:
        df = web.DataReader(name=tickerName, data_source='yahoo', start='10-01-2012', end='11-14-2018')
        df['Date'] = df.index
        df = df.asfreq('W-Mon', method='pad')

    mainHurst(df, tickerName)

    # print(df)

    # exit()

    # Date,    Open,    High,    Low,    Close, Adj Close, Volume    #

    # # Convert Date Format
    # df.columns = ['date','close','open','high','low']
    # df['date'] = pd.to_datetime(df['date'])
    #
    # # df = pd.DataFrame(np.linspace(0,100,101))
    # # df.columns = ['close']
    #
    # ma1 = movingAverage(df.close,window = 5)
    # ma2 = movingAverage(df.close,window = 21)
    # ma3 = movingAverage(df.close,window = 43)
    # # print(ma.ma)
    #
    # closes = Scatter(x=df.index,y=df.close,mode='markers')
    # ma1 = Scatter(x=ma1.index,y=ma1.movingAverage,mode='lines')
    # ma2 = Scatter(x=ma2.index, y=ma2.movingAverage, mode='lines')
    # ma3 = Scatter(x=ma3.index, y=ma3.movingAverage, mode='lines')

    # mask = (df.Date > '1997-12-31') & (df.Date <= '2003-07-07')
    # df = df.loc[mask]


def mainHurst(df, tickerName=''):
    df = df.reset_index(drop=True)

    df.columns = ['Date', 'Close', 'Open', 'High', 'Low']

    timeTolerance = 0.4  # 40%

    M54_weeks = 234  # approx 234 weeks in 54 months
    M54_weeksUpper = (1 + timeTolerance) * M54_weeks
    M54_weeksLower = (1 - timeTolerance) * M54_weeks

    M18_weeks = 78  # approx 78 weeks in 18 months
    M18_weeksUpper = (1 + timeTolerance) * M18_weeks
    M18_weeksLower = (1 - timeTolerance) * M18_weeks

    M9_weeks = 39  # approx 78 weeks in 9 months
    M9_weeksUpper = (1 + timeTolerance) * M9_weeks
    M9_weeksLower = (1 - timeTolerance) * M9_weeks

    W20_weeks = 26
    W10_weeks = 13

    # region:trendlines

    # dfCopy = df.copy()
    # dfCopy.columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
    # dfCopy['INDEX'] = dfCopy.index
    # if 'lowFirst' not in dfCopy.columns: dfCopy['lowFirst'] = dfCopy.open < dfCopy.close
    # #
    # # print(dfCopy)
    # # print()
    # # print(df)
    # #
    # # exit()
    #
    # minTrend, intTrend, majTrend = getTrendLine(dfCopy)
    #
    # minTrend = minTrend.merge(right=dfCopy, left_on='date', right_on='date')
    # intTrend = intTrend.merge(right=dfCopy, left_on='date', right_on='date')
    # majTrend = majTrend.merge(right=dfC opy, left_on='date', right_on='date')
    # endregion

    # print(minTrend)
    # # print(df.Date)
    # # print(dfCopy.date)
    # exit()

    period0 = 99
    period1 = 41  # periods below 41 removed, above 41 kept.
    period2 = 21
    period3 = 11
    period4 = 5

    # ma0 = movingAverage(df.Close, window=period0)
    ma1 = movingAverage(df.Close, window=period1)
    ma2 = movingAverage(df.Close, window=period2)
    ma3 = movingAverage(df.Close, window=period3, win_type='hann')
    ma4 = movingAverage(df.Close, window=period4, win_type='hann')

    # ma3 = movingAverage(df.Low, window=41)
    # taking MA and envelope of Close or Low doesn't make much difference)

    # ma0 = extendMA(ma0, period0)
    ma1 = extendMA(ma1, period=period1)
    ma2 = extendMA(ma2, period2)
    ma3 = extendMA(ma3, period3)
    ma4 = extendMA(ma4, period4)

    # allowMargin higher, bands get wider
    # upperband0, lowerband0 = envelopeFinder(df, ma0, allowMargin=0.1)
    upperband1, lowerband1 = envelopeFinder(df, ma1, allowMargin=0.1)
    upperband2, lowerband2 = envelopeFinder(df, ma2, allowMargin=0.1)
    upperband3, lowerband3 = envelopeFinder(df, ma3, allowMargin=0.1)
    upperband4, lowerband4 = envelopeFinder(df, ma4, allowMargin=0.1)

    # ma0Lp = lowestPointFinder(lowerband0)
    ma1Lp = lowestPointFinder(lowerband1)
    ma2Lp = lowestPointFinder(lowerband2)
    ma3Lp = lowestPointFinder(lowerband3)
    ma4Lp = lowestPointFinder(lowerband4)

    # ma0Lp = pd.DataFrame(ma0Lp)
    # ma0Lp.columns = ['point']
    ma1Lp = pd.DataFrame(ma1Lp)
    ma1Lp.columns = ['point']
    ma2Lp = pd.DataFrame(ma2Lp)
    ma2Lp.columns = ['point']
    ma3Lp = pd.DataFrame(ma3Lp)
    ma3Lp.columns = ['point']
    ma4Lp = pd.DataFrame(ma4Lp)
    ma4Lp.columns = ['point']

    # find cycles
    cyclesM54 = []
    cyclesM18 = []
    cyclesM9 = []
    cyclesW20 = []
    cyclesW10 = []

    # perhaps
    perhapsM54 = []
    perhapsM18 = []
    perhapsM9 = []
    perhapsW20 = []
    perhapsW10 = []

    # check 54M cycle. Any obvious ones.
    # sortedCloses
    # sortedCloses = df.sort_values('Close')
    lastPoint = lowerband1.index[-1] - M54_weeksLower
    tempHolder = []
    for tryPoint1 in ma1Lp.index:
        if tryPoint1 > lastPoint:
            # print('skip')
            continue

        for tryPoint2 in ma1Lp.index:
            distance = tryPoint2 - tryPoint1
            if (distance < M54_weeksUpper) and (distance > M54_weeksLower):
                # print('found')
                perhapsM54.append(tryPoint1)
                perhapsM54.append(tryPoint2)

    inverse = inverseMa(ma1.movingAverage, df.Close)

    ### check 18M cycle.
    # ma3Lp = lowestPointFinder(lowerband3)
    # majTrend.set_index('INDEX', inplace=True)
    # majTrend = majTrend.point

    dfLowLp = lowestPointFinder(df.Low)
    dfLowLp1Deg = lowestPointFinder1Deg(df.Low)

    dfLowLp = pd.DataFrame(dfLowLp)
    dfLowLp.columns = ['point']

    dfLowLp1Deg = pd.DataFrame(dfLowLp1Deg)
    dfLowLp1Deg.columns = ['point']

    # print(dfLowLp1Deg)
    #
    # for index,row in dfLowLp1Deg.iterrows():
    #     print(index, row.point)
    #
    # exit()

    perhaps, lpDiffList = getPotentials(ma1Lp, dfLowLp, period1, upperband1, lowerband1)

    for each in perhaps:
        if each[0] in cyclesM18:
            perhaps.remove(each)

    # starting point
    if len(lpDiffList) > 0:
        lowestDiff = min(lpDiffList)
        for each in perhaps:
            if each[1] == lowestDiff:
                cyclesM18.append(each[0])
                # remove from perhaps
                perhaps.remove(each)
                # fill down
                cyclesW10.append(each[0])
                cyclesW20.append(each[0])
                cyclesM9.append(each[0])

    if len(cyclesM18) < 1:
        print('------------ NOT SUITABLE -----------')
        return [1], [], []

    anchorPoint = Scatter(name='Starting Point', x=[cyclesM18[-1]], y=[0], mode='markers',
                          marker=dict(color='red', size=15))

    perhaps = checkLeftRight(cyclesM18[-1], perhaps, cyclesM18, M18_weeks, 1, timeTolerance, len(df),
                             [cyclesM9, cyclesW20, cyclesW10])
    perhaps = checkLeftRight(cyclesM18[-1], perhaps, cyclesM18, M18_weeks, 2, timeTolerance, len(df),
                             [cyclesM9, cyclesW20, cyclesW10])

    # region: old M18 code
    # if len(lpDiffList) > 0:
    #     lowestDiff = min(lpDiffList)
    #     for i in range(len(perhaps)):
    #         if perhaps[i][1] == lowestDiff:
    #             cyclesM18.append(perhaps[i][0])
    #
    #     # exit if more than one obvious one
    #     if len(cyclesM18) != 1: return 1
    #
    #     #fill down
    #     value = cyclesM18[0]
    #     cyclesW10.append(value)
    #     cyclesW20.append(value)
    #     cyclesM9.append(value)
    #
    #     # attempt 1: check 18M to the right
    #
    #
    #     if (cyclesM18[0] + M18_weeks) < len(df):
    #         lowerbound = cyclesM18[0] + M18_weeksLower
    #         upperbound = cyclesM18[0] + M18_weeksUpper
    #
    #         potential = (ma1Lp.point>lowerbound) & (ma1Lp.point<upperbound)
    #
    #         # if potential.sum() == 1:
    #             #TODO:add this point to cyclesM18 and fill down
    #
    #     #attempt 2: check 18M to the left
    #     if (cyclesM18[0] > M18_weeks):
    #         lowerbound = cyclesM18[0] - M18_weeksUpper
    #         upperbound = cyclesM18[0] - M18_weeksLower
    #
    #         potential = (ma1Lp.point > lowerbound) & (ma1Lp.point < upperbound)
    #
    #         # if potential.sum() == 1:
    #         # TODO:add this point to cyclesM18 and fill down
    # endregion

    ### check 9M cycle

    perhaps, lpDiffList = getPotentials(ma2Lp, dfLowLp, period2, upperband2, lowerband2, bufferFactor=0.5)

    for each in perhaps:
        if each[0] in cyclesM9:
            perhaps.remove(each)

    # print('1:',perhaps)

    for weekNum in cyclesM18:
        # print("<<< M9 >>>>>:", weekNum)
        # print()

        # attempt 1: check to the right

        # region: check 1 period to right and left
        perhaps = checkLeftRight(weekNum, perhaps, cyclesM9, M9_weeks, numCycles=1, cycleAllowance=timeTolerance,
                                 totalDataLength=len(df), filldownList=[cyclesW20, cyclesW10])
        # print()
        # print('After check 1 period:', perhaps)
        # endregion

        # region: check 2 periods to right and left
        perhaps = checkLeftRight(weekNum, perhaps, cyclesM9, M9_weeks, numCycles=2, cycleAllowance=timeTolerance,
                                 totalDataLength=len(df), filldownList=[cyclesW20, cyclesW10, cyclesM18],
                                 upperCycles=[cyclesM18], upperCyclePeriods=[M18_weeks])
        # print()
        # print('After check 2 periods:',perhaps)
        # endregion

    # check 20W cycle
    perhaps, lpDiffList = getPotentials(ma3Lp, dfLowLp, period3, upperband3, lowerband3, bufferFactor=0.5)

    for each in perhaps:
        if each[0] in cyclesW20:
            perhaps.remove(each)

    for weekNum in cyclesM9:
        # print("<<< W20 >>>>>:", weekNum)
        # print()

        # region: check 1 period to right and left
        perhaps = checkLeftRight(weekNum, perhaps, cyclesW20, W20_weeks, numCycles=1, cycleAllowance=timeTolerance,
                                 totalDataLength=len(df), filldownList=[cyclesW10])
        # print()
        # print('After check 1 period:', perhaps)
        # endregion

        # region: check 2 period to right and left
        perhaps = checkLeftRight(weekNum, perhaps, cyclesW20, W20_weeks, numCycles=2, cycleAllowance=timeTolerance,
                                 totalDataLength=len(df), filldownList=[cyclesW10, cyclesM9],
                                 upperCycles=[cyclesM9, cyclesM18], upperCyclePeriods=[M9_weeks, M18_weeks])
        # print()
        # print('After check 2 periods:',perhaps)
        # endregion

    # check 10W cycle
    perhaps, lpDiffList = getPotentials(ma4Lp, dfLowLp, period4, upperband4, lowerband4, bufferFactor=0.5,
                                        envFactor=0.5)

    # print('Week 10')
    # print(perhaps)
    # exit()
    for each in perhaps:
        if each[0] in cyclesW10:
            perhaps.remove(each)

    for weekNum in cyclesW20:
        # print("<<< W10 >>>>>:", weekNum)
        # print()

        # region: check 1 period to right and left
        perhaps = checkLeftRight(weekNum, perhaps, cyclesW10, W10_weeks, numCycles=1, cycleAllowance=timeTolerance,
                                 totalDataLength=len(df), filldownList=[])
        # print()
        # print('After check 1 period:', perhaps)
        # endregion

        # region: check 2 period to right and left
        perhaps = checkLeftRight(weekNum, perhaps, cyclesW10, W10_weeks, numCycles=2, cycleAllowance=timeTolerance,
                                 totalDataLength=len(df), filldownList=[cyclesW20],
                                 upperCycles=[cyclesW20, cyclesM9, cyclesM18],
                                 upperCyclePeriods=[W20_weeks, M9_weeks, M18_weeks])
        # print()
        # print('After check 2 periods:',perhaps)
        # endregion

    # print('% of W10 points compared to total length divided by 10:', len(cyclesW10) * 100 / (len(df) / 10), '%')
    GAUGE.append(len(cyclesW10) * 100 / (len(df) / 10))
    # if len(cyclesW10) < ( len(df)/10): return 2

    if len(cyclesW10) * 100 / (len(df) / 10) < 10:
        print('--------- DATA IS NOT CLEAR ENOUGH ------')
        return [2],[], []

    gapfillers(cyclesW10, dfLowLp1Deg, dfLowLp, ma4Lp, len(df), timeTolerance, nominal=W10_weeks)

    cyclesW20 = set(cyclesW20)
    cyclesW20 = list(cyclesW20)

    cyclesW10 = set(cyclesW10)
    cyclesW10 = list(cyclesW10)

    cyclesM9 = set(cyclesM9)
    cyclesM9 = list(cyclesM9)

    cyclesM18 = set(cyclesM18)
    cyclesM18 = list(cyclesM18)

    fillUp(longerPeriodCycle=cyclesW20, shorterPeriodCycle=cyclesW10, nominal=W20_weeks, lastPoint=len(df),
           timeTolerance=timeTolerance)
    fillUp(longerPeriodCycle=cyclesM9, shorterPeriodCycle=cyclesW20, nominal=M9_weeks, lastPoint=len(df),
           timeTolerance=timeTolerance)
    fillUp(longerPeriodCycle=cyclesM18, shorterPeriodCycle=cyclesM9, nominal=M18_weeks, lastPoint=len(df),
           timeTolerance=timeTolerance)

    # remove duplicates
    cyclesM54 = removeDuplicates(cyclesM54)
    cyclesM18 = removeDuplicates(cyclesM18)
    cyclesM9 = removeDuplicates(cyclesM9)
    cyclesW20 = removeDuplicates(cyclesW20)
    cyclesW10 = removeDuplicates(cyclesW10)

    postDownFiller(cyclesM18, cyclesM9, dfLowLp, dfLowLp1Deg, fillDown=[cyclesW20, cyclesW10])
    postDownFiller(cyclesM9, cyclesW20, dfLowLp, dfLowLp1Deg, fillDown=[cyclesW10])
    postDownFiller(cyclesW20, cyclesW10, dfLowLp, dfLowLp1Deg)

    filterRatios(cyclesM54, cyclesM18, cyclesM9, cyclesW20, cyclesW10, df.Low)

    # fill in the top most level
    cyclesM18 = removeDuplicates(cyclesM18)

    fillUpTopLevel(prevCycle=removeDuplicates(cyclesM18), topCycle=cyclesM54,
                   compareList=[ma1Lp, ma2Lp, ma3Lp, ma4Lp, dfLowLp], prevPeriod=period1, ma=ma4)

    # remove duplicates
    cyclesM54 = removeDuplicates(cyclesM54)
    cyclesM18 = removeDuplicates(cyclesM18)
    cyclesM9 = removeDuplicates(cyclesM9)
    cyclesW20 = removeDuplicates(cyclesW20)
    cyclesW10 = removeDuplicates(cyclesW10)

    avg0, M54Avg = getAvgPeriods(cyclesM54, cyclesM18[0], cyclesM18[-1])
    avg1, M18Avg = getAvgPeriods(cyclesM18, cyclesM18[0], cyclesM18[-1])
    avg2, M9Avg = getAvgPeriods(cyclesM9, cyclesM18[0], cyclesM18[-1])
    avg3, W20Avg = getAvgPeriods(cyclesW20, cyclesM18[0], cyclesM18[-1])
    avg4, W10Avg = getAvgPeriods(cyclesW10, cyclesM18[0], cyclesM18[-1])

    avgList = [
        avg0, avg1, avg2, avg3, avg4

    ]

    avgListFloat = [ M54Avg, M18Avg, M9Avg, W20Avg, W10Avg

    ]

    # plot
    boeing = Ohlc(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, text=df.Date,
                  increasing=dict(line=dict(color='black')),
                  decreasing=dict(line=dict(color='black')),
                  )

    # ma0 = Scatter(x=ma0.index, y=ma0.movingAverage, mode='lines', marker=dict(color='orange'), line=dict(dash='dash'))
    # upperband0 = Scatter(x=upperband0.index, y=upperband0, mode='lines', marker=dict(color='orange'))
    # lowerband0 = Scatter(name = 'period '+str(period0)+' lowerband',x=lowerband0.index, y=lowerband0, mode='lines', marker=dict(color='orange'))

    ma1 = Scatter(x=ma1.index, y=ma1.movingAverage, mode='lines', marker=dict(color='blue'), line=dict(dash='dash'))
    upperband1 = Scatter(x=upperband1.index, y=upperband1, mode='lines', marker=dict(color='blue'))
    lowerband1 = Scatter(name='period ' + str(period1) + ' lowerband', x=lowerband1.index, y=lowerband1, mode='lines',
                         marker=dict(color='blue'))

    ma2 = Scatter(x=ma2.index, y=ma2.movingAverage, mode='lines', marker=dict(color='red'), line=dict(dash='dash'))
    upperband2 = Scatter(x=upperband2.index, y=upperband2, mode='lines', marker=dict(color='red'))
    lowerband2 = Scatter(name='period ' + str(period2) + ' lowerband', x=lowerband2.index, y=lowerband2, mode='lines',
                         marker=dict(color='red'))

    ma3 = Scatter(x=ma3.index, y=ma3.movingAverage, mode='lines', marker=dict(color='green'), line=dict(dash='dash'))
    upperband3 = Scatter(x=upperband3.index, y=upperband3, mode='lines', marker=dict(color='green'))
    lowerband3 = Scatter(name='period ' + str(period3) + ' lowerband', x=lowerband3.index, y=lowerband3, mode='lines',
                         marker=dict(color='green'))

    ma4 = Scatter(x=ma4.index, y=ma4.movingAverage, mode='lines', marker=dict(color='green'), line=dict(dash='dash'))
    upperband4 = Scatter(x=upperband4.index, y=upperband4, mode='lines', marker=dict(color='green'))
    lowerband4 = Scatter(name='period ' + str(period4) + ' lowerband', x=lowerband4.index, y=lowerband4, mode='lines',
                         marker=dict(color='green'))

    # ma0Lp = Scatter(x=ma0Lp.index, y=ma0Lp.point, mode='markers', marker=dict(color='orange', size=10))
    ma1Lp = Scatter(x=ma1Lp.index, y=ma1Lp.point, mode='markers', marker=dict(color='navy', size=10))
    ma2Lp = Scatter(x=ma2Lp.index, y=ma2Lp.point, mode='markers', marker=dict(color='red', size=10))
    ma3Lp = Scatter(x=ma3Lp.index, y=ma3Lp.point, mode='markers', marker=dict(color='green', size=10))
    ma4Lp = Scatter(x=ma4Lp.index, y=ma4Lp.point, mode='markers', marker=dict(color='black', size=10))

    M54 = Scatter(name=avgList[0], x=cyclesM54, y=[0] * len(cyclesM54), mode='markers',
                  marker=dict(size=10, color='green'), showlegend=True)
    M18 = Scatter(name=avgList[1], x=cyclesM18, y=[0] * len(cyclesM18), mode='markers',
                  marker=dict(size=10, color='orange'), showlegend=True)
    M9 = Scatter(name=avgList[2], x=cyclesM9, y=[0] * len(cyclesM9), mode='markers', marker=dict(size=10, color='navy'),
                 showlegend=True)
    W20 = Scatter(name=avgList[3], x=cyclesW20, y=[0] * len(cyclesW20), mode='markers',
                  marker=dict(size=10, color='blue'), showlegend=True)
    W10 = Scatter(name=avgList[4], x=cyclesW10, y=[0] * len(cyclesW10), mode='markers',
                  marker=dict(size=10, color='purple'), showlegend=True)

    # minTrend = Scatter(x=minTrend.INDEX, y=minTrend.point, mode='lines', line=dict(color='grey'))
    # intTrend = Scatter(x=intTrend.INDEX, y=intTrend.point, mode='lines', line=dict(color='orange'))
    # majTrend = Scatter(x=majTrend.index, y=majTrend, mode='lines', line=dict(color='purple'))
    dfLowLp = Scatter(x=dfLowLp.index, y=dfLowLp.point, mode='markers', line=dict(color='purple'),
                      marker=dict(size=7))

    dfLowLp1Deg = Scatter(x=dfLowLp1Deg.index, y=dfLowLp1Deg.point, mode='markers', line=dict(color='orange'),
                          marker=dict(size=10))

    trace = [
        boeing,
        # ma0,
        # lowerband0,
        # ma0Lp,
        ma1,
        upperband1,
        lowerband1,
        ma1Lp,
        ma2,
        upperband2,
        lowerband2,
        ma2Lp,
        ma3,
        upperband3,
        lowerband3,
        ma3Lp,
        #      fit,
        # lows,
        # minTrend, intTrend,majTrend,
        # lowerband4,
        # ma4Lp,

        dfLowLp1Deg,
        dfLowLp,

    ]

    layout = Layout(xaxis=dict(rangeslider=dict(visible=False)))
    figure = Figure(trace, layout=layout)

    # plotly.offline.plot(figure)

    # plotVerticalSubs('hurst_'+tickerName+'.html',trace, title=tickerName,lastIdx=len(df), avgList = avgList, others=[inverse,M54, M18, M9, W20, W10],anchorPoint=anchorPoint)

    # print(perhapsM54)
    # print(perhapsM18)
    # print(perhapsM9)
    # print(perhapsW20)
    # print(perhapsW10)

    # get projected dates
    M54LastDate = df.iloc[cyclesM54[-1]].Date
    M18LastDate = df.iloc[cyclesM18[-1]].Date
    M9LastDate = df.iloc[cyclesM9[-1]].Date
    W20LastDate = df.iloc[cyclesW20[-1]].Date
    W10LastDate = df.iloc[cyclesW10[-1]].Date

    M54projs = []
    M18projs = []
    M9projs = []
    W20projs = []
    W10projs = []

    M54projs.append(M54LastDate + timedelta(weeks=M54Avg))

    M18projs.append(M18LastDate + timedelta(weeks=M18Avg))
    for i in range(30):
        nextDate = M18projs[-1] + timedelta(weeks=M18Avg)

        if nextDate <= M54projs[-1]:
            M18projs.append(nextDate)
        else:
            break

    M9projs.append(M9LastDate + timedelta(weeks=M9Avg))
    for i in range(30):
        nextDate = M9projs[-1] + timedelta(weeks=M9Avg)

        if nextDate <= M54projs[-1]:
            M9projs.append(nextDate)
        else:
            break

    W20projs.append(W20LastDate + timedelta(weeks=W20Avg))
    for i in range(30):
        nextDate = W20projs[-1] + timedelta(weeks=W20Avg)

        if nextDate <= M54projs[-1]:
            W20projs.append(nextDate)
        else:
            break

    W10projs.append(W10LastDate + timedelta(weeks=W10Avg))
    for i in range(30):
        nextDate = W10projs[-1] + timedelta(weeks=W10Avg)

        if nextDate <= M54projs[-1]:
            W10projs.append(nextDate)
        else:
            break

    # print('Avg Periods')
    # print('Expected:', M54_weeks, 'IRL:', avgList[0])
    # print('Expected:', M18_weeks, 'IRL:', avgList[1])
    # print('Expected:', M9_weeks, 'IRL:', avgList[2])
    # print('Expected:', W20_weeks, 'IRL:', avgList[3])
    # print('Expected:', W10_weeks, 'IRL:', avgList[4])

    return [M54projs, M18projs, M9projs, W20projs, W10projs], avgListFloat, M54LastDate


if __name__ == '__main__':

    start = time.time()

    # tickerList = ['BA','EURUSD',
    #     '^GSPC','^DJI','^IXIC','^NYA','^XAX','^BUK100P','^RUT','^VIX',
    #               # '^FTSE',
    #               '^GDAXI','^FCHI',
    #               # '^STOXX50E', #######
    #               '^N100','^BFX',
    #               # 'IMOEX.ME',
    #               '^N225',
    #     # '^HSI',
    #     '000001.SS',
    #     # '^STI',
    #               '^AXJO','^AORD','^BSESN','^JKSE',
    #     # '^KLSE',
    #     '^NZ50','^KS11','^TWII','^GSPTSE','^BVSP',
    #     # '^MXX ',
    #               '^IPSA','^MERV','^TA125.TA',
    #     # '^CASE30',
    # '^JN0U.JO']

    # tickerList = ['^NYA','^XAX','^BUK100P','^RUT','^GDAXI','^FCHI']

    # tickerList = ['BA']
    # tickerList = ['^NYA']
    # tickerList = ['^IPSA']
    tickerList = ['EURUSD']

    for each in tickerList:
    #     print()
    #     print('Working on Ticker:', each)
    #     # print(main(each))
    #     try:
        main(each)
        # except Exception as e:
        #     print('*** Ticker', each, 'failed ***')
        #     print(e)
        #     print()
        #
        # exit()

    end = time.time()

    timeTaken = (end - start)
    print('time taken:', timeTaken)

    print(GAUGE)
    print(np.mean(GAUGE))

    # region: test remove extras
    # size = 100
    # # x = np.linspace(0,size,10)
    # y = np.random.rand(size)
    #
    # lows = pd.DataFrame(y,columns=['Low'])
    #
    #
    #
    # A = [0,20,80,size]
    # B = [10,20,24,37,68,77,80,81]
    #
    # removeExtras(2,A,B,lows.Low)
    #
    # print(A)
    # print(B)
    # endregion

    # region: polyfit stuff
    # z = np.polyfit(x=df.index, y=df.Low, deg=70)
    # p = np.poly1d(z)
    #
    # fit = Scatter(x=df.index, y=p(df.index), mode='lines', marker=dict(color='orange'),line=dict(width=3))
    # lows = Scatter(x=df.index, y=df.Low, mode='lines', marker=dict(color='grey'),line=dict(width=3))
    # endregion

    # region: FFT stuff
    # Ts = 81
    # Fs = 1.0/Ts
    # y = df.copy().Low
    #
    #
    #
    # n = len(y)  # length of the signal
    # k = np.arange(n)
    # T = n / Fs
    # frq = k / T  # two sides frequency range
    # frq = frq[range(int(n / 2))]  # one side frequency range
    #
    # Y = np.fft.fft(y) / n  # fft computing and normalization
    # Y = Y[range(int(n / 2))]
    #
    #
    # fftTrace = Scatter(x=frq,y=abs(Y),mode='markers')
    # figure=Figure([fftTrace])
    # plotly.offline.plot(figure_or_data=figure)
    # endregion