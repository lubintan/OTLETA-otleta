from math import pi
from datetime import datetime
from datetime import timedelta

import pandas as pd
import numpy as np
import plotly
from plotly.graph_objs import Layout, Scatter, Line, Ohlc, Figure, Histogram, Bar, Table
import plotly.figure_factory as ff
import scipy as sp
import time
from trendFunctions import *

def getPotentials(maLp,dfLowLp,period,upperband,lowerband,bufferFactor=0.2):
    # maLp = maLp.copy()
    # maLp = pd.DataFrame(maLp)
    # maLp.columns = ['point']

    buffer = bufferFactor * period
    envWidth = upperband.dropna().iloc[0] - lowerband.dropna().iloc[0]
    lpDiffList = []
    perhaps = []

    for i1, r1 in maLp.iterrows():
        for i2, r2 in dfLowLp.iterrows():
            if (abs(i1 - i2) < buffer) and (r2.point < (r1.point + (0.1 * envWidth))):
                perhaps.append([i2, (r2.point - r1.point)])
                lpDiffList.append(r2.point - r1.point)
                break

    return perhaps,lpDiffList

def inverseMa(ma,data):

    inverse = data - ma


    return Bar(name='inverse MA',x=inverse.index,y=inverse,)

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


def envelopeFinder(df,ma,allowMargin = 0.1):
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

def movingAverage(data, window):
    ma = data.rolling(window=window,center=True,).mean()
    ma = pd.DataFrame(ma)
    ma.columns=['movingAverage']
    ma['original'] = data
    # ma = ma.dropna()

    return ma

def extendMA(ma,period):
    noNA = ma.dropna()
    point2 = noNA.iloc[-1].movingAverage
    point1 = noNA.iloc[-2].movingAverage

    index2 = noNA.index[-1]
    index1 = noNA.index[-2]

    #y = mx+c
    gradient = point2 - point1 #because the x2-x1 here is 1.
    #c = y-mx
    yIntercept = point1 - (index1 * gradient)

    halfPeriod=int(period/2)
    # print(halfPeriod)
    # exit()
    for index,row in ma.iloc[-1*halfPeriod::].iterrows():
        row.movingAverage = gradient * index + yIntercept


    # print(ma.tail())
    return ma

def plotVerticalSubs(mainTrace,others=[]):
    numRows = len(others)
    fig = plotly.tools.make_subplots(
        rows=numRows+1, cols=1, shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.001,row_width=[0.1]*numRows+[0.8])

    for eachTrace in mainTrace:
        fig.append_trace(eachTrace,1,1)



    for i in range(numRows):
        fig.append_trace(others[i], i + 2, 1)

    fig['layout'].update(xaxis=dict(rangeslider=dict(visible=False),gridwidth=2,mirror='ticks',nticks=30,showgrid=True),
                         yaxis2 = dict(showticklabels=False,title='Inverse MA'),
                         yaxis3=dict(showticklabels=False,title='54M'),
                         yaxis4=dict(showticklabels=False,title='18M'),
                         yaxis5=dict(showticklabels=False,title='9M'),
                         yaxis6=dict(showticklabels=False,title='20W'),
                         yaxis7=dict(showticklabels=False,title='10W'),
    )



    plotly.offline.plot(fig)

def main():
    window = 5
    # df = pd.read_excel('EURUSD Weekly Data for Swing Indicator.xlsx')[0:100]

    df = pd.read_csv('BA.csv')
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

    mask = (df.Date > '1997-12-31') & (df.Date <= '2003-07-07')
    # mask = df.Date==df.Date
    df = df.loc[mask]

    df.reset_index(drop=True, inplace=True)

    boeing = Ohlc(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, text=df.Date,
                  increasing=dict(line=dict(color='black')),
                  decreasing=dict(line=dict(color='black')),
                  )

    timeTolerance = 0.4  # 40%

    M54_weeks = 234  # approx 234 weeks in 54 months
    M54_weeksUpper = (1 + timeTolerance) * M54_weeks
    M54_weeksLower = (1 - timeTolerance) * M54_weeks

    M18_weeks = 78 # approx 78 weeks in 18 months
    M18_weeksUpper = (1 + timeTolerance) * M18_weeks
    M18_weeksLower = (1 - timeTolerance) * M18_weeks

    M9_weeks = 39 # approx 78 weeks in 9 months
    M9_weeksUpper = (1 + timeTolerance) * M9_weeks
    M9_weeksLower = (1 - timeTolerance) * M9_weeks


    # trendlines

    dfCopy = df.copy()
    dfCopy.columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']
    dfCopy['INDEX'] = dfCopy.index
    if 'lowFirst' not in dfCopy.columns: dfCopy['lowFirst'] = dfCopy.open < dfCopy.close
    #
    # print(dfCopy)
    # print()
    # print(df)
    #
    # exit()

    minTrend, intTrend, majTrend = getTrendLine(dfCopy)

    minTrend = minTrend.merge(right=dfCopy, left_on='date', right_on='date')
    intTrend = intTrend.merge(right=dfCopy, left_on='date', right_on='date')
    majTrend = majTrend.merge(right=dfCopy, left_on='date', right_on='date')

    # print(minTrend)
    # # print(df.Date)
    # # print(dfCopy.date)
    # exit()

    period1 = 41
    period2 = 21
    period3 = 11

    ma1 = movingAverage(df.Close, window=period1)
    ma2 = movingAverage(df.Close, window=period2)
    ma3 = movingAverage(df.Close, window=period3)
    # ma3 = movingAverage(df.Low, window=41)
    # taking MA and envelope of Close or Low doesn't make much difference)

    # ma1 = extendMA(ma1,period = period1)
    # ma2 = extendMA(ma2, period2)
    ma3 = extendMA(ma3, period3)

    # allowMargin higher, bands get wider
    upperband1, lowerband1 = envelopeFinder(df, ma1, allowMargin=0.1)
    upperband2, lowerband2 = envelopeFinder(df, ma2, allowMargin=0.1)
    upperband3, lowerband3 = envelopeFinder(df, ma3, allowMargin=0.1)

    ma1Lp = lowestPointFinder(lowerband1)
    ma2Lp = lowestPointFinder(lowerband2)
    ma3Lp = lowestPointFinder(lowerband3)
    
    ma1Lp = pd.DataFrame(ma1Lp)
    ma1Lp.columns = ['point']
    ma2Lp = pd.DataFrame(ma2Lp)
    ma2Lp.columns = ['point']
    ma3Lp = pd.DataFrame(ma3Lp)
    ma3Lp.columns = ['point']

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

    inverse = inverseMa(ma1.movingAverage,df.Close)

    ### check 18M cycle.
    # ma3Lp = lowestPointFinder(lowerband3)
    majTrend.set_index('INDEX', inplace=True)
    majTrend = majTrend.point
    dfLowLp = lowestPointFinder(df.Low)

    dfLowLp = pd.DataFrame(dfLowLp)
    dfLowLp.columns = ['point']

    perhaps , lpDiffList = getPotentials(ma1Lp,dfLowLp,period1,upperband1,lowerband1)

    # for each in perhaps:
    #     perhapsM18.append(each)

    if len(lpDiffList) > 0:
        lowestDiff = min(lpDiffList)
        for i in range(len(perhaps)):
            if perhaps[i][1] == lowestDiff:
                cyclesM18.append(perhaps[i][0])

        # exit if more than one obvious one
        if len(cyclesM18) != 1: return 1

        #fill down
        value = cyclesM18[0]
        cyclesW10.append(value)
        cyclesW20.append(value)
        cyclesM9.append(value)

        # attempt 1: check 18M to the right


        if (cyclesM18[0] + M18_weeks) < len(df):
            lowerbound = cyclesM18[0] + M18_weeksLower
            upperbound = cyclesM18[0] + M18_weeksUpper

            potential = (ma1Lp.point>lowerbound) & (ma1Lp.point<upperbound)

            # if potential.sum() == 1:
                #TODO:add this point to cyclesM18 and fill down

        #attempt 2: check 18M to the left
        if (cyclesM18[0] > M18_weeks):
            lowerbound = cyclesM18[0] - M18_weeksUpper
            upperbound = cyclesM18[0] - M18_weeksLower

            potential = (ma1Lp.point > lowerbound) & (ma1Lp.point < upperbound)

            # if potential.sum() == 1:
            # TODO:add this point to cyclesM18 and fill down


    ### check 9M cycle

    perhaps, lpDiffList = getPotentials(ma2Lp, dfLowLp, period2, upperband2, lowerband2, bufferFactor=0.5)

    for each in perhaps:
        if each[0] in cyclesM9:
            perhaps.remove(each)


    print(perhaps)


    for weekNum in cyclesM18:
    #attempt 1: check to the right

    #region: check 1 period to right and left
        if (weekNum + M9_weeks) < len(df):
            lowerbound = weekNum + M9_weeksLower
            upperbound = weekNum + M9_weeksUpper

            perhaps2 = []
            lpDiffList = []
            for i in range(len(perhaps)):
                week = perhaps[i]
                if (week[0] < upperbound) and (week[0] > lowerbound):
                    perhaps2.append(week)
                    lpDiffList.append(week[1])

            if len(perhaps2) > 1:
                lowestDiff = min(lpDiffList)

                for each in perhaps2:
                    if each[1] == lowestDiff:
                        cyclesM9.append(each[0])
                        # fill down
                        value = cyclesM9[-1]
                        cyclesW20.append(value)
                        cyclesW10.append(value)

                        #remove from perhaps
                        for each in perhaps:
                            if each[0] == value:
                                perhaps.remove(each)

            elif len(perhaps2) > 0:
                cyclesM9.append(perhaps2[-1][0])
                # fill down
                value = cyclesM9[-1]
                cyclesW20.append(value)
                cyclesW10.append(value)

                # remove from perhaps
                for each in perhaps:
                    if each[0] == value:
                        perhaps.remove(each)

    if (weekNum > M9_weeks):
            lowerbound = weekNum - M9_weeksUpper
            upperbound = weekNum - M9_weeksLower


            perhaps2 = []
            lpDiffList = []
            for i in range(len(perhaps)):
                week = perhaps[i]
                if (week[0] < upperbound) and (week[0] > lowerbound):
                    perhaps2.append(week)
                    lpDiffList.append(week[1])

            if len(perhaps2) > 1:
                lowestDiff = min(lpDiffList)

                for each in perhaps2:
                    if each[1] == lowestDiff:
                        cyclesM9.append(each[0])
                        # fill down
                        value = cyclesM9[-1]
                        cyclesW20.append(value)
                        cyclesW10.append(value)

                        #remove from perhaps
                        for each in perhaps:
                            if each[0] == value:
                                perhaps.remove(each)

            elif len(perhaps2) > 0:
                cyclesM9.append(perhaps2[-1][0])
                # fill down
                value = cyclesM9[-1]
                cyclesW20.append(value)
                cyclesW10.append(value)

                # remove from perhaps
                for each in perhaps:
                    if each[0] == value:
                        perhaps.remove(each)

    #endregion

    print()
    print(perhaps)

    #region: check 2 period to right and left

    if (weekNum + (M9_weeks*2)) < len(df):
        lowerbound = weekNum + M9_weeks + M9_weeksLower
        upperbound = weekNum + M9_weeks + M9_weeksUpper

        print('upper',upperbound)
        print('lower', lowerbound)

        perhaps2 = []
        lpDiffList = []
        for i in range(len(perhaps)):
            week = perhaps[i]
            if (week[0] < upperbound) and (week[0] > lowerbound):
                perhaps2.append(week)
                lpDiffList.append(week[1])

        if len(perhaps2) > 1:
            lowestDiff = min(lpDiffList)

            for each in perhaps2:
                if each[1] == lowestDiff:
                    cyclesM9.append(each[0])
                    # fill down
                    value = cyclesM9[-1]
                    cyclesW20.append(value)
                    cyclesW10.append(value)

                    # remove from perhaps
                    for each in perhaps:
                        if each[0] == value:
                            perhaps.remove(each)


        elif len(perhaps2) > 0:
            cyclesM9.append(perhaps2[-1][0])
            # fill down
            value = cyclesM9[-1]
            cyclesW20.append(value)
            cyclesW10.append(value)

            # remove from perhaps
            for each in perhaps:
                if each[0] == value:
                    perhaps.remove(each)

    if (weekNum > (2*M9_weeks)):
        lowerbound = weekNum - M9_weeks - M9_weeksUpper
        upperbound = weekNum - M9_weeks - M9_weeksLower

        perhaps2 = []
        lpDiffList = []
        for i in range(len(perhaps)):
            week = perhaps[i]
            if (week[0] < upperbound) and (week[0] > lowerbound):
                perhaps2.append(week)
                lpDiffList.append(week[1])

        if len(perhaps2) > 1:
            lowestDiff = min(lpDiffList)

            for each in perhaps2:
                if each[1] == lowestDiff:
                    cyclesM9.append(each[0])
                    # fill down
                    value = cyclesM9[-1]
                    cyclesW20.append(value)
                    cyclesW10.append(value)

                    # remove from perhaps
                    for each in perhaps:
                        if each[0] == value:
                            perhaps.remove(each)


        elif len(perhaps2) > 0:
            cyclesM9.append(perhaps2[-1][0])
            # fill down
            value = cyclesM9[-1]
            cyclesW20.append(value)
            cyclesW10.append(value)

            # remove from perhaps
            for each in perhaps:
                if each[0] == value:
                    perhaps.remove(each)

    #endregion

    print()
    print(perhaps)

    # plot
    ma1 = Scatter(x=ma1.index, y=ma1.movingAverage, mode='lines', marker=dict(color='blue'), line=dict(dash='dash'))
    upperband1 = Scatter(x=upperband1.index, y=upperband1, mode='lines', marker=dict(color='blue'))
    lowerband1 = Scatter(x=lowerband1.index, y=lowerband1, mode='lines', marker=dict(color='blue'))

    ma2 = Scatter(x=ma2.index, y=ma2.movingAverage, mode='lines', marker=dict(color='red'), line=dict(dash='dash'))
    upperband2 = Scatter(x=upperband2.index, y=upperband2, mode='lines', marker=dict(color='red'))
    lowerband2 = Scatter(x=lowerband2.index, y=lowerband2, mode='lines', marker=dict(color='red'))

    ma3 = Scatter(x=ma3.index, y=ma3.movingAverage, mode='lines', marker=dict(color='green'), line=dict(dash='dash'))
    upperband3 = Scatter(x=upperband3.index, y=upperband3, mode='lines', marker=dict(color='green'))
    lowerband3 = Scatter(x=lowerband3.index, y=lowerband3, mode='lines', marker=dict(color='green'))

    ma1Lp = Scatter(x=ma1Lp.index, y=ma1Lp.point, mode='markers', marker=dict(color='navy', size=10))
    ma2Lp = Scatter(x=ma2Lp.index, y=ma2Lp.point, mode='markers', marker=dict(color='olive', size=10))
    ma3Lp = Scatter(x=ma3Lp.index, y=ma3Lp.point, mode='markers', marker=dict(color='olive', size=10))

    M54 = Scatter(x=cyclesM54, y=[0] * len(cyclesM54), mode='markers', marker=dict(size=10, color='navy'))
    M18 = Scatter(x=cyclesM18, y=[0] * len(cyclesM18), mode='markers', marker=dict(size=10, color='navy'))
    M9 = Scatter(x=cyclesM9, y=[0] * len(cyclesM9), mode='markers', marker=dict(size=10, color='navy'))
    W20 = Scatter(x=cyclesW20, y=[0] * len(cyclesW20), mode='markers', marker=dict(size=10, color='navy'))
    W10 = Scatter(x=cyclesW10, y=[0] * len(cyclesW10), mode='markers', marker=dict(size=10, color='navy'))

    minTrend = Scatter(x=minTrend.INDEX, y=minTrend.point, mode='lines', line=dict(color='grey'))
    intTrend = Scatter(x=intTrend.INDEX, y=intTrend.point, mode='lines', line=dict(color='orange'))
    majTrend = Scatter(x=majTrend.index, y=majTrend, mode='lines', line=dict(color='purple'))
    dfLowLp = Scatter(x=dfLowLp.index, y=dfLowLp.point, mode='markers', line=dict(color='purple'),
                         marker=dict(size=7))

    trace = [
        boeing,
        # ma1, upperband1, lowerband1,
        # ma2, upperband2,
        # lowerband2,
        # ma2Lp,
         ma3,upperband3,
        lowerband3,
        #      fit,
        # lows,
        # ma1Lp,
        # minTrend, intTrend,majTrend,
        dfLowLp,
        ma3Lp
    ]

    layout = Layout(xaxis=dict(rangeslider=dict(visible=False)))
    figure = Figure(trace, layout=layout)


    plotVerticalSubs(trace, others=[inverse,M54, M18, M9, W20, W10])

    print(perhapsM54)
    print(perhapsM18)
    print(perhapsM9)
    print(perhapsW20)
    print(perhapsW10)

if __name__ == '__main__':

    main()




    #region: polyfit stuff
    # z = np.polyfit(x=df.index, y=df.Low, deg=70)
    # p = np.poly1d(z)
    #
    # fit = Scatter(x=df.index, y=p(df.index), mode='lines', marker=dict(color='orange'),line=dict(width=3))
    # lows = Scatter(x=df.index, y=df.Low, mode='lines', marker=dict(color='grey'),line=dict(width=3))
    #endregion

    #region: FFT stuff
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
    #endregion