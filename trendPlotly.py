from math import pi
from datetime import datetime
from datetime import timedelta

import pandas as pd
import numpy as np
import plotly
from plotly.graph_objs import Layout,Scatter,Line,Ohlc,Figure,Histogram,Bar
import plotly.figure_factory as ff
import scipy as sp
import time

startTime = time.time()

#TODO:
#plotly
#table and chart subplot
#figure factory subplot
# gantt chart for intersecting projections (basic)
# mixed subplot
# tables, histograms

#TODO:
# Hurst Cycles - Start Date
# Window 1999-2001
# Check for toppest top and bottomest bottom
# Take date - Get year, month, week, day for that top/bottom point.
# If top & bottom:
# Look at next year. If top taken out, choose bottomest bottom date. And vice versa.
# If cannot resolve, go to following year and see if top or bottom taken out.
#
#TODO:
# Price Retracements
# All on the right hand side.
# See email for retracement levels.
# For now, for each full trend (up or down).
# "mini" projections only for current, where don't know whether trend has ended or not.
# Differnet levels for up trend or down trend.
#
#
# TODO: UPDATE BY EACH DAY/BAR!
# Current trend (always up or down.)
# See if broke previous top or bottom.
# If sideways, (trend up/down will just keep flipping.)
# If doesn't break, keep the previous trend.
#
#
#TODO:
# Tops and Bottoms Projection
# Find latest trend (up or down), can be current trend also.
# collate H-H, H-L, L-L, L-H data. (for both uptrend and downtrend)
# Project for each H-H, H-L, L-L, L-H using previous set of numbers from whole data set.
# For each H-H, H-L, L-L, L-H, show historgram of dates vs 1x of each duration from previous trend. Do for both tops and bottoms.
# if current trend has no eg. H-L data yet, use previous trend's data.
# replace previous set of data, once there is 1 value from current trend.
# IMPT: For H-L and L-L, only project ONCE for each set of numbers.
# Combine H-L and L-L charts (to see which dates have hits).
# Combine L-H and H-H charts (to see which dates have hits).
# 2 things to look for: 1. highest frequecnies of intervals from the past. 2. projections - where the dates line up.
# to project Lows, use L-L, and H-L (different starting points)
# to project Highs, use H-H, and L-H (different starting points)
# See if can use gantt chart to show which projections come from where.

#TODO: Lost Motion
#TODO: Signal Tops/Bottoms

#TODO:
# - DONE: split 3 trendlines
# - DONE: outside bar include closer to open/close thing for highs and lows
# - DONE: HH, HL, LH, LL - histogram?

#TODO:
#     - write program to be efficient: only make changes to new data collected (don’t recompute everything.)

# - DONE: see swing high and low numbers. High green circle, low red circle.

# TITLE = 'TrendLine Delay = ' + str(DELAY)
TITLE = 'minor-grey, intermediate-blue, major-black'

def check3Points(counter,topsAndBottoms,trendUp,topOrBottom):
    if not (
        (topsAndBottoms.iloc[counter].trendUp==trendUp) and
        (topsAndBottoms.iloc[counter-1].trendUp == trendUp) and
        (topsAndBottoms.iloc[counter-2].trendUp == trendUp) and
        (topsAndBottoms.iloc[counter].top == topOrBottom) and
        (topsAndBottoms.iloc[counter-1].top == ~topOrBottom) and
        (topsAndBottoms.iloc[counter-2].top == topOrBottom)
        ):
        return False, counter-1

    diff = topsAndBottoms.iloc[counter].date - topsAndBottoms.iloc[counter-2].date

    return diff.days, counter-2


def trendProjector(topsAndBottoms):

    # print(topsAndBottoms)

    # diff = topsAndBottoms.iloc[-1].date - topsAndBottoms.iloc[-2].date
    # print(diff.days)
    # print(topsAndBottoms.iloc[-2].date + timedelta(days=diff.days))
    # print(topsAndBottoms.iloc[-2].date + timedelta(days=diff.days*2))
    # exit()

    currentTrendUp = topsAndBottoms.iloc[-1].trendUp
    currentTopOrBottom = topsAndBottoms.iloc[-1].top
    currentDate = topsAndBottoms.iloc[-1].date

    projectionLimit = 365*2 #days

    timeDiffs = []
    counter = -1
    while True:
        diffInDays, counter = check3Points(counter,topsAndBottoms,currentTrendUp,currentTopOrBottom)

        if (diffInDays==False):
            if len(timeDiffs) > 1:
                break
        else:
            timeDiffs.append(diffInDays)

    print(timeDiffs)

    dates=[]

    for diffInDays in timeDiffs:
        multiplier = 1
        nextDate = currentDate+timedelta(days=diffInDays*multiplier)
        while (nextDate - currentDate).days <=projectionLimit:
            dates.append(nextDate)
            multiplier +=1
            nextDate = currentDate + timedelta(days=diffInDays * multiplier)

    # print(len(dates)dates)

    cycles = pd.DataFrame(dates,columns=['date'])
    # print(cycles)
    cycles = cycles['date'].value_counts()
    # print(cycles.size)
    cycles = pd.DataFrame(cycles)
    # print()
    # exit()

    # print(cycles.index)

    trendText = 'Trend Up'
    topBottomText = 'Top to Top'
    if not currentTrendUp: trendText='Trend Down'
    if not currentTopOrBottom: topBottomText = 'Bottom to Bottom'

    cyclesTrace = Bar(x=cycles.index, y=cycles.date,
                       # xbins=dict(start=np.min(HH_bars), size=size, end=np.max(HH_bars)),
                         showlegend=False,
                       # hoverinfo='none',
                      name='Projection for '+trendText+', '+topBottomText,
                       )

    fig = Figure(data=[cyclesTrace])

    return plotly.offline.plot(figure_or_data=fig,
                               show_link=False,
                               output_type='div',
                               include_plotlyjs=False,
                               # filename='minorHLData.html',
                               auto_open=False,
                               config={'displaylogo': False,
                                       'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'zoomIn2d',
                                                                  'zoomOut2d',
                                                                  'resetScale2d', 'hoverCompareCartesian', 'lasso2d'],
                                       'displayModeBar': False
                                       })




def trendFinder(stuff):
    topsAndBottoms = stuff['topsAndBottoms']

    barHeight = 0.03
    # barBias =

    trendUp = pd.DataFrame()
    trendUp['date'] = topsAndBottoms.date
    topsAndBottoms['trendUp'] = (topsAndBottoms.point > topsAndBottoms.shift(2).point)
    trendUp['trendUp'] = (topsAndBottoms.point > topsAndBottoms.shift(2).point) * barHeight *2 - barHeight
    trendUp.reset_index(drop=True,inplace=True)

    trendUp.ix[0,'trendUp'] = 0
    trendUp.ix[1, 'trendUp'] = 0

    ups = trendUp[trendUp.trendUp>0]
    downs = trendUp[trendUp.trendUp<0]
    # print(ups)
    # print(downs)

    upsTrace = Bar(name='Trend Up', x=ups.date,y=ups.trendUp,marker=dict(color='green'),
                   text='up',textposition='auto',opacity=0.7,
                   showlegend=False,hoverinfo="text")
    downsTrace = Bar(name='Trend Down', x=downs.date, y=downs.trendUp, marker=dict(color='red'),
                     text='down',textposition='auto',opacity=0.7,
                     showlegend=False, hoverinfo='text')

    # topsAndBottomsTrace,tops,bots = plotTrendlines(topsAndBottoms,stuff,name='Test',color='blue', width=2)

    # fig = plotly.tools.make_subplots(rows=2, cols=1, )
    # fig.append_trace([upsTrace,downsTrace], 1, 1)
    # fig.append_trace(LL_barsTrace, 1, 2)



    # fig = Figure([upsTrace,
    #               downsTrace,
    #               # topsAndBottomsTrace,tops,bots
    #               ],
    #              layout=Layout(xaxis=dict(showgrid=True))
    #              )
    # plotly.offline.plot(fig)


    return upsTrace, downsTrace, topsAndBottoms

def plotTopBotHist(stuff):
    HH_bars = stuff['HH_bars']
    HH_barsMean = np.mean(HH_bars)
    HH_barsMode = sp.stats.mode(HH_bars).mode[0]
    HH_title = 'Average TOP to TOP duration: %.2f' % (HH_barsMean) + ' bars'
    HH_bars = HH_bars.value_counts()
    HH_barsTrace = Bar(x=HH_bars.index, y=HH_bars.values,
                       # xbins=dict(start=np.min(HH_bars), size=size, end=np.max(HH_bars)),
                         showlegend=False,
                       hoverinfo='none',
                       )

    LL_bars = stuff['LL_bars']
    LL_barsMean = np.mean(LL_bars)
    LL_barsMode = sp.stats.mode(LL_bars).mode[0]
    LL_title = 'Average BOTTOM to BOTTOM duration: %.2f' % (LL_barsMean) + ' bars'
    LL_bars = LL_bars.value_counts()
    LL_barsTrace = Bar(x=LL_bars.index, y=LL_bars.values,
                       # xbins=dict(start=np.min(HH_bars), size=size, end=np.max(HH_bars)),
                         showlegend=False,
                       hoverinfo='none',
                       )

    HL_bars = stuff['HL_bars']
    HL_barsMean = np.mean(HL_bars)
    HL_barsMode = sp.stats.mode(HL_bars).mode[0]
    HL_title = 'Average TOP to BOTTOM duration: %.2f' % (HL_barsMean) + ' bars'
    HL_bars = HL_bars.value_counts()
    HL_barsTrace = Bar(x=HL_bars.index, y=HL_bars.values,
                       # xbins=dict(start=np.min(HH_bars), size=size, end=np.max(HH_bars)),
                         showlegend=False,
                       hoverinfo='none',
                       )

    LH_bars = stuff['LH_bars']
    LH_barsMean = np.mean(LH_bars)
    LH_barsMode = sp.stats.mode(LH_bars).mode[0]
    LH_title = 'Average BOTTOM to TOP duration: %.2f' % (LH_barsMean) + ' bars'
    LH_bars = LH_bars.value_counts()
    LH_barsTrace = Bar(x=LH_bars.index, y=LH_bars.values,
                       # xbins=dict(start=np.min(HH_bars), size=size, end=np.max(HH_bars)),
                         showlegend=False,
                       hoverinfo='none',
                       )

    fig = plotly.tools.make_subplots(rows=2, cols=2, subplot_titles=(HH_title, LL_title, HL_title, LH_title))
    fig.append_trace(HH_barsTrace, 1, 1)
    fig.append_trace(LL_barsTrace, 1, 2)
    fig.append_trace(HL_barsTrace, 2, 1)
    fig.append_trace(LH_barsTrace, 2, 2)




    fig['layout'].update(title='Top and Bottom Duration Data',
                         bargap=0.2,
                         # dtick=0.25, ticklen=8, tickwidth=4, tickcolor='red',
                         xaxis={'title':'Duration in Bars','dtick':1} ,yaxis={'title':'Frequency count','dtick':1},
                         xaxis2={'title': 'Duration in Bars', 'dtick': 1},
                         yaxis2={'title': 'Frequency count', 'dtick': 1},
                         xaxis3={'title': 'Duration in Bars', 'dtick': 1},
                         yaxis3={'title': 'Frequency count', 'dtick': 1},
                         xaxis4={'title': 'Duration in Bars', 'dtick': 1},
                         yaxis4={'title': 'Frequency count', 'dtick': 1},

                         )


    # print(fig)

    return plotly.offline.plot(figure_or_data=fig,
                        show_link=False,
                        output_type='div',
                        include_plotlyjs=False,
                        # filename='minorHLData.html',
                        auto_open=False,
                        config={'displaylogo': False,
                                'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'zoomIn2d', 'zoomOut2d',
                                                           'resetScale2d', 'hoverCompareCartesian', 'lasso2d'],
                                'displayModeBar': False
                                })

def groupByMonths(df):
    months = pd.DataFrame()
    months['month'] = df['date'].dt.month
    months['year'] = df['date'].dt.year


    months = months.drop_duplicates().reset_index(drop=True)
    months['high'] = np.nan
    months['low'] = np.nan
    months['open'] = np.nan
    months['close'] = np.nan
    months['lowFirst'] = np.True_


    for index,row in months.iterrows():
        thisMonth = (df.date.dt.month==row.month) & (df.date.dt.year==row.year)
        monthlyData = df[thisMonth]

        maxPrice = monthlyData.loc[monthlyData['high'].idxmax()]
        minPrice = monthlyData.loc[monthlyData['low'].idxmin()]

        earliest = monthlyData.loc[monthlyData['date'].idxmin()]
        latest = monthlyData.loc[monthlyData['date'].idxmax()]

        months.at[index,'high'] = maxPrice.high
        months.at[index,'low'] = minPrice.low
        months.at[index,'open'] = earliest.open
        months.at[index,'close'] = latest.close

        if maxPrice.date < minPrice.date:
            months.at[index,'lowFirst'] = False

    months['date'] = months.apply(lambda row: datetime(
        row['year'], row['month'], 1), axis=1)


    return pd.DataFrame(months)

def plotter(figure, filename, HL_html):
    plotly.offline.plot(figure_or_data=figure,
                        show_link=False,
                        output_type='file',
                        include_plotlyjs=False,
                        filename=filename,
                        auto_open=False,
                        config={'displaylogo': False,
                                'modeBarButtonsToRemove':['sendDataToCloud','select2d','zoomIn2d','zoomOut2d',
                                                          'resetScale2d','hoverCompareCartesian','lasso2d']})
    line = '<script src="plotly-latest.min.js"></script>'
    with open(filename, 'r+') as htmlFile:
        contents = htmlFile.read()
        htmlFile.seek(0, 0)
        htmlFile.write(line.rstrip('\r\n') + '\n' + contents + '\n' + HL_html)

def plotTrendlines(trendLine, stuff, name, color, width, dash=None):
    line = Scatter(name=name+ 'Trend', x=trendLine.date, y=trendLine.point,
                    mode='lines+markers',
                    line=dict(color=color,
                              width=width,
                              dash=dash
                              ),
                    hoverinfo='none',
                   showlegend=False,

                    )

    tops = Scatter(name=name+' Tops', x=stuff['tops'].date, y=stuff['tops'].point,
                        mode='markers+text',
                        line=dict(color=color,
                                  width=width,
                                  dash=dash
                                  ),
                        marker=dict(symbol='circle'),
                        hoverinfo='none',
                        text=stuff['topText'],
                        textposition='top center',
                        textfont=dict(color='green')
                        )

    bottoms = Scatter(name=name+' Bottoms', x=stuff['bottoms'].date, y=stuff['bottoms'].point,
                       mode='markers+text',
                       line=dict(color=color,
                                 width=width,
                                 dash=dash
                                 ),
                        marker=dict(symbol='circle'),
                       hoverinfo='none',
                       text=stuff['bottomText'],
                       textposition='bottom center',
                       textfont=dict(color='red')
                       )

    return line,tops,bottoms

def getTrendTopsAndBottoms(trendLine, df):

    tops = ((trendLine.point>trendLine.shift(1).point) & (trendLine.point>trendLine.shift(-1).point))
    bottoms = ((trendLine.point < trendLine.shift(1).point) & (trendLine.point < trendLine.shift(-1).point))

    reindexed = trendLine.reset_index(drop=True)
    dateIndexOnly = pd.DataFrame()
    dateIndexOnly['i'] = df.reset_index(drop=True).index
    dateIndexOnly['date'] = df.date
    reindexed = reindexed.merge(dateIndexOnly, on='date')

    topPoints = pd.DataFrame(reindexed[tops])
    # topPoints['i'] = topPoints.i
    bottomPoints = pd.DataFrame(reindexed[bottoms])
    # bottomPoints['i'] = bottomPoints.i
    topAndBottomPoints = pd.DataFrame(reindexed[tops|bottoms])
    # topAndBottomPoints['i'] = topAndBottomPoints.i
    

    # region: High to High
    HH_time = topPoints.date - topPoints.shift(1).date
    HH_price = topPoints.point - topPoints.shift(1).point
    HH_bars = topPoints.i - topPoints.shift(1).i

    HH_time = HH_time.dropna()
    HH_price = HH_price.dropna()

    HH_bars = HH_bars.dropna()

    # fig = ff.create_distplot([HH_bars], group_labels=['test1'],
    #endregion

    #region: Low to Low
    LL_time = bottomPoints.date - bottomPoints.shift(1).date
    LL_price = bottomPoints.point - bottomPoints.shift(1).point
    LL_bars = bottomPoints.i - bottomPoints.shift(1).i

    LL_time = LL_time.dropna()
    LL_price = LL_price.dropna()
    LL_bars = LL_bars.dropna()
    
    LL_barsMean = np.mean(LL_bars)
    LL_barsMode = sp.stats.mode(LL_bars).mode[0]

    LL_barsTrace = Histogram(x=LL_bars, xbins=dict(start=np.min(LL_bars), size=1, end=np.max(LL_bars)))
    LL_barsFig = Figure(data=[LL_barsTrace])
    #endregion


    #region: High to Low and Low to High
    mixed_time = topAndBottomPoints.date - topAndBottomPoints.shift(1).date
    mixed_price = topAndBottomPoints.point - topAndBottomPoints.shift(1).point
    mixed_bars = topAndBottomPoints.i - topAndBottomPoints.shift(1).i

    mixed_time = mixed_time.dropna().reset_index(drop=True)
    mixed_price = mixed_price.dropna().reset_index(drop=True)
    mixed_bars = mixed_bars.dropna().reset_index(drop=True)

    topAndBottomPoints.reset_index(drop=True,inplace=True)

    if topAndBottomPoints.iloc[0].point < topAndBottomPoints.iloc[1].point: #bottom first
        HL_time = mixed_time[mixed_time.index%2==0]#even
        LH_time = mixed_time[mixed_time.index%2==1]#odd

        HL_price = mixed_price[mixed_price.index % 2 == 0]  
        LH_price = mixed_price[mixed_price.index % 2 == 1]  

        HL_bars = mixed_bars[mixed_bars.index % 2 == 0]  
        LH_bars = mixed_bars[mixed_bars.index % 2 == 1]  

    else:
        HL_time = mixed_time[mixed_time.index % 2 == 1]
        LH_time = mixed_time[mixed_time.index % 2 == 0]

        HL_price = mixed_price[mixed_price.index % 2 == 1]
        LH_price = mixed_price[mixed_price.index % 2 == 0]

        HL_bars = mixed_bars[mixed_bars.index % 2 == 1]
        LH_bars = mixed_bars[mixed_bars.index % 2 == 0]



    #endregion

    trendLine['top'] = tops
    trendLine['bottom'] = bottoms
    topText = []
    bottomText = []
    for index, row in trendLine.iterrows():
        if row.top:
            topText.append(str(row.point))
            # bottomText.append('')
        elif row.bottom:
            # topText.append('')
            bottomText.append(str(row.point))
        # else:
        #     topText.append('')
        #     bottomText.append('')

    # print(trendLine)
    # print(len(trendLine[tops]))
    # print(len(topText.))
    # print(topText)
    # exit()


    return dict(tops=trendLine[tops],
                bottoms=trendLine[bottoms],
                topsAndBottoms=trendLine[tops|bottoms],
                topText = topText,
                bottomText=bottomText,
                HH_bars=HH_bars,
                LL_bars=LL_bars,
                HL_bars=HL_bars,
                LH_bars=LH_bars,
                )

def checkPrevPoints(dfIgnoreInsideBars, index, row, DELAY):
    # takes out the chunk to be checked and reverses the order
    testDf = (dfIgnoreInsideBars[(index - DELAY):index]).iloc[::-1]



    # check for consecutive lower points
    runningLow = row.low
    trendLow = True
    for idx, entry in testDf.iterrows():
        if runningLow < entry.low:
            trendLow = True
            runningLow = entry.low
        else:
            trendLow = False
            break

    # check for consecutive higher points
    runningHigh = row.high
    trendHigh = True
    for idx, entry in testDf.iterrows():
        if runningHigh > entry.high:
            trendHigh = True
            runningHigh = entry.high
        else:
            trendHigh = False
            break

    # add points if necessary

    if trendLow:
        return [row.date, row.low]
    elif trendHigh:
        return [row.date, row.high]

    # otherwise do nothing

        return None

def processOutsideBars(row, trendPoints, DELAY,minorPoints):
    if len(minorPoints) >= 2:
        if minorPoints[-1][1] >= minorPoints[-2][1]:  # trending up
            if not row.lowFirst:  # high first
                if DELAY == 1:  # high then low
                    trendPoints.append([row.date, row.high])
                    trendPoints.append([row.date, row.low])
                else:  # 1 bar up
                    trendPoints.append([row.date, row.high])
            else:  # low first
                if DELAY == 1:  # low then high
                    trendPoints.append([row.date, row.low])
                    trendPoints.append([row.date, row.high])
                else:  # 1 bar up
                    trendPoints.append([row.date, row.high])

        elif minorPoints[-1][1] < minorPoints[-2][1]:  # trending down

            if not row.lowFirst:  # high first
                if DELAY == 1:  # high then low
                    trendPoints.append([row.date, row.high])
                    trendPoints.append([row.date, row.low])
                else:  # 1 bar up
                    trendPoints.append([row.date, row.high])
            else:  # low first
                if DELAY == 1:  # low then high
                    trendPoints.append([row.date, row.low])
                    trendPoints.append([row.date, row.high])
                else:  # 1 bar down
                    trendPoints.append([row.date, row.low])
    return trendPoints

def getTrendLine(dfIgnoreInsideBars):
    minorPoints = []
    intermediatePoints = []
    majorPoints = []

    # dfIgnoreInsideBars = pd.DataFrame(df)

    for index, row in dfIgnoreInsideBars.iterrows():
        if index < 1: continue
        if row.outside:

            if index >=2: intermediatePoints = processOutsideBars(row,intermediatePoints,DELAY=2,minorPoints=minorPoints)
            if index >=3: majorPoints = processOutsideBars(row,majorPoints,DELAY=3,minorPoints=minorPoints)
            minorPoints = processOutsideBars(row, minorPoints, DELAY=1, minorPoints=minorPoints)
            continue

        # minor points
        result = checkPrevPoints(dfIgnoreInsideBars, index, row, DELAY=1)
        if result!=None: minorPoints.append(result)

        # intermediate points
        if index >=2:
            result = checkPrevPoints(dfIgnoreInsideBars, index, row, DELAY=2)
            if result != None: intermediatePoints.append(result)

        # major points
        if index >=3:
            result = checkPrevPoints(dfIgnoreInsideBars, index, row, DELAY=3)
            if result != None: majorPoints.append(result)

    trendLine1 = pd.DataFrame(minorPoints, columns=['date', 'point'])
    trendLine2 = pd.DataFrame(intermediatePoints, columns=['date', 'point'])
    trendLine3 = pd.DataFrame(majorPoints, columns=['date', 'point'])

    return trendLine1, trendLine2, trendLine3


if __name__ == '__main__':

    df = pd.read_excel('EURUSD Weekly Data for Swing Indicator.xlsx')
    # df = pd.read_csv('EURUSD Weekly Data for Swing Indicator.csv')

    # df = pd.read_excel('test1.xlsx')



    # Convert Date Format
    df.columns = ['date','close','open','high','low']
    df['date'] = pd.to_datetime(df['date'])

    # df = groupByMonths(df)

    # region: add lowFirst column if no timing data
    if 'lowFirst' not in df.columns: df['lowFirst'] = df.open < df.close
    # endregion


#inside bars (fully ignore for trend line calculation)
#region: INSIDE BARS

    activeDate=[]
    activeClose=[]
    activeOpen=[]
    activeHigh=[]
    activeLow=[]
    activeLowFirst=[]
    
    insideDate=[]
    insideClose=[]
    insideOpen=[]
    insideHigh=[]
    insideLow=[]

    for i,row in df.iterrows():

        if i==0:
            activeDate.append(row.date)
            activeClose.append(row.close)
            activeOpen.append(row.open)
            activeHigh.append(row.high)
            activeLow.append(row.low)
            activeLowFirst.append(row.lowFirst)
            continue


        if (activeHigh[-1]>row.high) & (activeLow[-1]<row.low):
            insideDate.append(row.date)
            insideClose.append(row.close)
            insideOpen.append(row.open)
            insideHigh.append(row.high)
            insideLow.append(row.low)
            continue

        activeDate.append(row.date)
        activeClose.append(row.close)
        activeOpen.append(row.open)
        activeHigh.append(row.high)
        activeLow.append(row.low)
        activeLowFirst.append(row.lowFirst)

    # dfIgnoreInsideBars = pd.DataFrame(columns=['date', 'close', 'open', 'high', 'low'])

    noInsideBars = {
                    'date':activeDate,
                    'close':activeClose,
                    'open':activeOpen,
                    'high':activeHigh,
                    'low':activeLow,
                    'lowFirst':activeLowFirst,
                    }
    insideBarsOnly = {
                    'date': insideDate,
                    'close': insideClose,
                    'open': insideOpen,
                    'high': insideHigh,
                    'low': insideLow,
    }
    dfIgnoreInsideBars = pd.DataFrame.from_dict(noInsideBars)
    dfInsideBarsOnly = pd.DataFrame.from_dict(insideBarsOnly)

    # print(dfIgnoreInsideBars)
    # exit()
    # dfIgnoreInsideBars = pd.concat([df[:DELAY], dfIgnoreInsideBars])
    # dfIgnoreInsideBars = dfIgnoreInsideBars.reset_index(drop=True)

#endregion INSIDE BARS


#region: OUTSIDE BARS
    dfIgnoreInsideBars['outside'] = (dfIgnoreInsideBars.high>dfIgnoreInsideBars.shift(1).high) & (dfIgnoreInsideBars.low<dfIgnoreInsideBars.shift(1).low)


    # print(dfIgnoreInsideBars)
    # exit()

#endregion

    trendLine1, trendLine2, trendLine3 = getTrendLine(dfIgnoreInsideBars)



#### TRENDLINE PROCESSING  END #####

    # topAndBottomPoints, HH_bars, LL_bars, HL_bars, LH_bars
    minorStuff = getTrendTopsAndBottoms(trendLine1,df)
    intermediateStuff = getTrendTopsAndBottoms(trendLine2,df)
    majorStuff = getTrendTopsAndBottoms(trendLine3,df)

    # plot H L data
    minorHL_html = plotTopBotHist(minorStuff)
    intermediateHL_html = plotTopBotHist(intermediateStuff)
    majorHL_html = plotTopBotHist(majorStuff)

    # trends
    minorUps, minorDowns, minorTopsBottoms = trendFinder(minorStuff)
    intermediateUps, intermediateDowns, intermediateTopsBottoms = trendFinder(intermediateStuff)
    majorUps, majorDowns, majorTopsBottoms = trendFinder(majorStuff)

    # projections
    minorHL_html = trendProjector(minorTopsBottoms)
    intermediateHL_html = trendProjector(intermediateTopsBottoms)
    majorHL_html = trendProjector(majorTopsBottoms)


    insideBars = Ohlc(name='Inside Bars',x=dfInsideBarsOnly.date,open=dfInsideBarsOnly.open,close=dfInsideBarsOnly.close,
                    high=dfInsideBarsOnly.high,low=dfInsideBarsOnly.low,
                    opacity=0.5,
                    line=dict(width=1),
                   hoverinfo='none',
                   # increasing=dict(line=dict(color= '#17BECF')),
                   # decreasing=dict(line=dict(color= '#17BECF')),
                   )

    activeBars = Ohlc(name='Active Bars',x=dfIgnoreInsideBars.date,open=dfIgnoreInsideBars.open,close=dfIgnoreInsideBars.close,high=dfIgnoreInsideBars.high,low=dfIgnoreInsideBars.low,
                    opacity=1,
                    line=dict(width=2.5),
                      hoverinfo='none',
                    hoverlabel=dict(namelengthsrc='none'),
                      # showlegend=False,
                   # increasing=dict(line=dict(color= '#17BECF')),
                   # decreasing=dict(line=dict(color= '#17BECF')),

                   )

    dfOutsideHighFirst = dfIgnoreInsideBars[dfIgnoreInsideBars['outside'] & (~dfIgnoreInsideBars['lowFirst'])]
    dfOutsideLowFirst = dfIgnoreInsideBars[dfIgnoreInsideBars['outside'] & dfIgnoreInsideBars['lowFirst']]

    OHF = Ohlc(name='Outside High First', x=dfOutsideHighFirst.date, open=dfOutsideHighFirst.open,
                  close=dfOutsideHighFirst.close, high=dfOutsideHighFirst.high, low=dfOutsideHighFirst.low,
                  opacity=1,
                  line=dict(width=4),
                  increasing=dict(line=dict(color= 'orange')),
                  decreasing=dict(line=dict(color= 'orange')),
                  hoverinfo='none',
                      )

    OLF = Ohlc(name='Outside Low First', x=dfOutsideLowFirst.date, open=dfOutsideLowFirst.open,
               close=dfOutsideLowFirst.close, high=dfOutsideLowFirst.high, low=dfOutsideLowFirst.low,
               opacity=1,
               line=dict(width=4),
               increasing=dict(line=dict(color= 'purple')),
               decreasing=dict(line=dict(color= 'purple')),
               hoverinfo='none',
               )
    # plot minor trendline points and lines
    minor, minorTops, minorBottoms = plotTrendlines(trendLine1,minorStuff,name='Minor', color='grey',width=2)

    # plot intermediate trendline points and lines
    intermediate, intermediateTops, intermediateBottoms = plotTrendlines(trendLine2, intermediateStuff,
                                                                         name='Intermediate', color='#000080', width=2)

    # plot major trendline points and lines
    major, majorTops, majorBottoms = plotTrendlines(trendLine3, majorStuff, name='Major', color='black', width=2)





    minorData = [insideBars, activeBars, OHF, OLF, minor, minorTops, minorBottoms,minorUps,minorDowns]
    intermediateData = [insideBars, activeBars, OHF, OLF, intermediate, intermediateTops, intermediateBottoms,intermediateUps, intermediateDowns]
    majorData = [insideBars, activeBars, OHF, OLF, major, majorTops, majorBottoms,majorUps, majorDowns]


    layout = Layout(
        title = 'EUR/USD Weekly',
        xaxis=dict(
            rangeslider=dict(
                visible=False
            ),
            showgrid=True,
        ),
        showlegend=True,
        # annotations=minorAnnot
    )

    minorFig = Figure(data=minorData, layout=layout)
    intermediateFig = Figure(data=intermediateData, layout=layout)
    majorFig = Figure(data=majorData, layout=layout)

    # plotly.offline.plot(fig,image='png',image_filename='3trends',image_width=7200,image_height=1200)

    plotter(minorFig,'minorTrend_daily.html', minorHL_html)
    plotter(intermediateFig, 'intermediateTrend_daily.html', intermediateHL_html)
    plotter(majorFig, 'majorTrend_daily.html', majorHL_html)

    # help(plotly.offline.plot)

    endTime = time.time()
    elapsed = endTime-startTime
    print("Operation took a total of %.3f milliseconds." %(elapsed))