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


# TODO:
# plotly
# table and chart subplot
# figure factory subplot
# gantt chart for intersecting projections (basic)
# mixed subplot
# tables, histograms

# TODO:
# Window 1999-2001
# Check for toppest top and bottomest bottom
# Take date - Get year, month, week, day for that top/bottom point.
# If top & bottom:
# Look at next year. If top taken out, choose bottomest bottom date. And vice versa.
# If cannot resolve, go to following year and see if top or bottom taken out.
#
# TODO:
# Price Retracements
# All on the right hand side.
# See email for retracement levels.
# For now, for each full trend (up or down).
# "mini" projections only for current, where don't know whether trend has ended or not.
# Differnet levels for up trend or down trend.
#
#
# TODO: UPDATE BY EACH DAY/BAR!
# DONE: Current trend (always up or down.)
# DONE: See if broke previous top or bottom.
# DONE: If sideways, (trend up/down will just keep flipping.)
# DONE: If doesn't break, keep the previous trend.
#
# TODO:
# Fig 12.2, 12.3 - show difference in price, time at max/min points.
# (Non-trend indicator. Raw data)

#
# TODO:
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

# TODO: Lost Motion
# TODO: Signal Tops/Bottoms

# TODO:
# - DONE: split 3 trendlines
# - DONE: outside bar include closer to open/close thing for highs and lows
# - DONE: HH, HL, LH, LL - histogram?

# TODO:
#     - write program to be efficient: only make changes to new data collected (don’t recompute everything.)

# - DONE: see swing high and low numbers. High green circle, low red circle.

# TITLE = 'TrendLine Delay = ' + str(DELAY)
TITLE = 'minor-grey, intermediate-blue, major-black'

def hurstSines(lowestPoint,df, projLimitDate=None):
    lengthDf = len(df)
    hurstX=[]
    hurstXBehind=[]

    if projLimitDate==None:
        projLimit = 12  # bars
        projLimitDate = df.iloc[-1].date + timedelta(days=projLimit * 7)

    for i in range(lengthDf * 77):
        nextDate = lowestPoint + timedelta(days= i)

        if (nextDate > projLimitDate): break
        hurstX.append(nextDate)

    for i in range(1,lengthDf * 77):
        nextDate = lowestPoint - timedelta(days= i)

        if (nextDate < df.iloc[0].date): break
        hurstXBehind.append(nextDate)

    hurstXInt = np.arange(start=-1*len(hurstXBehind),stop=len(hurstX),step=1)
    hurstXBehind.reverse()
    hurstX = hurstXBehind + hurstX

    return hurstX, hurstXInt


def verticalPlot(mainTrace,others=[], others2=[],others3=[]):
    numRows = 1

    fig = plotly.tools.make_subplots(
        rows=numRows+1, cols=1, shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.1,row_width=[0.2]*numRows+[0.8])

    for eachTrace in mainTrace:
        fig.append_trace(eachTrace,1,1)

    # fig.append_trace(anchorPoint, 4, 1)
    for each in others:
        fig.append_trace(each, 2, 1)
    #
    # for each in others2:
    #     fig.append_trace(each, 3, 1)
    #
    # for each in others3:
    #     fig.append_trace(each, 4, 1)

    # for each in hurst2:
    #     fig.append_trace(each, 7, 1)
    # for each in hurst3:
    #     fig.append_trace(each, 8, 1)
    # for each in hurst4:
    #     fig.append_trace(each, 9, 1)
    # for each in hurst5:
    #     fig.append_trace(each, 10, 1)


    fig['layout'].update(barmode='stack',xaxis=dict(rangeslider=dict(visible=False),showgrid=True),
    #                      yaxis2 = dict(showticklabels=False,title='Inverse MA'),
    #                      yaxis3=dict(showticklabels=False,title='240W'),
    #                      yaxis4=dict(showticklabels=False,title='80W'),
    #                      yaxis5=dict(showticklabels=False,title='40W'),
    #                      yaxis6=dict(showticklabels=False,title='20W'),
    #                      yaxis7=dict(showticklabels=False,title='10W'),
    #                      title=title,
    #                      # annotations=[dict(
    #                      #                    x=lastIdx,
    #                      #                    y=0,
    #                      #                    xref='x3',
    #                      #                    yref='y3',
    #                      #                    text='Avg: '+ avgList[0] + 'weeks',
    #                      #                    showarrow=False,
    #                      #                    align='right',
    #                      #                ),
    #                      #                 dict(
    #                      #                     x=lastIdx,
    #                      #                     y=0,
    #                      #                     xref='x4',
    #                      #                     yref='y4',
    #                      #                     text='Avg: '+ avgList[1] + 'weeks',
    #                      #                     showarrow=False,
    #                      #                     align='right',
    #                      #                 ),
    #                      #                 dict(
    #                      #                     x=lastIdx,
    #                      #                     y=0,
    #                      #                     xref='x5',
    #                      #                     yref='y5',
    #                      #                     text='Avg: '+ avgList[2] + 'weeks',
    #                      #                     showarrow=False,
    #                      #                     align='right',
    #                      #                 ),
    #                      #                 dict(
    #                      #                     x=lastIdx,
    #                      #                     y=0,
    #                      #                     xref='x6',
    #                      #                     yref='y6',
    #                      #                     text='Avg: '+ avgList[3] + 'weeks',
    #                      #                     showarrow=False,
    #                      #                     align='right',
    #                      #                 ),
    #                      #                 dict(
    #                      #                     x=lastIdx,
    #                      #                     y=0,
    #                      #                     xref='x7',
    #                      #                     yref='y7',
    #                      #                     text='Avg: '+ avgList[4] + 'weeks',
    #                      #                     showarrow=False,
    #                      #                     align='right',
    #                      #                 ),]
                                    )


    plotly.offline.plot(fig)
    # plotly.offline.plot(fig,output_type='file',filename=filename)

def clusterAlgo2(levelList, close, startX, endX):

    # levelList = [[1,3,5],[3.5,7,9,11],[0.3,2,7],[5.5,4.5,10.1,3]]

    #find max spacing from min range (in case minspacing between sets is too big)
    maxSpace = 0
    for eachRange in levelList:
        rangeSpacings = []
        for i in range(len(eachRange)-1):
            rangeSpacings.append(abs(eachRange[i] - eachRange[i+1]))
        thisMaxSpacing = max(rangeSpacings)
        if thisMaxSpacing > maxSpace:
            maxSpace = thisMaxSpacing



    spaceFactor = 2
    minSpace = maxSpace #np.inf
    clusters = []
    length = len(levelList)

    for each in levelList:
        each.sort()

    # print(levelList)

    for i in range(length):

        for j in range(1,length):
            nextIdx = i + j
            if nextIdx >= length: break

            # prevDist = np.inf

            for mainVal in levelList[i]:
                for subVal in levelList[nextIdx]:
                    dist = abs(mainVal-subVal)
                    # if dist >= prevDist: break

                    if dist < minSpace:
                        minSpace = dist

                    # prevDist = dist

    finalClustersAbove = []
    finalClustersBelow = []
    finalAboveLevel = -1
    finalBelowLevel = -1
    
    
    if minSpace==maxSpace:
        # if no clusters found
        aboveMin = np.inf
        belowMin = np.inf
        
        for eachRange in levelList:
            for val in eachRange:
                aboveDist = val - close
                belowDist = close - val
                
                if (aboveDist>0) and (aboveDist<aboveMin):
                    finalAboveLevel = val
                    aboveMin = aboveDist
                    continue
                if (belowDist > 0) and (belowDist < belowMin):
                    finalBelowLevel = val
                    belowMin = belowDist


    else:#if clusters exist
        clusterCutoff = spaceFactor * minSpace
    
        for i in range(length):
            for j in range(1, length):
                nextIdx = i + j
                if nextIdx >= length: break
                for mainVal in levelList[i]:
                    for subVal in levelList[nextIdx]:
                        dist = abs(mainVal - subVal)
    
                        if dist <= clusterCutoff:
                            clusters.append([mainVal, subVal])
                            
        # find clusters above close and below close
        clustersAbove = {}
        clustersAboveDist = []
        clustersBelow = {}
        clustersBelowDist = []
        
        
        for each in clusters:
            avgDist = (sum(each) / len(each))
            if (avgDist > close):
                clustersAboveDist.append(avgDist)
                clustersAbove[avgDist] = each
                continue
            elif (avgDist < close):
                clustersBelowDist.append(avgDist)
                clustersBelow[avgDist] = each
                        
                
        if len(clustersAboveDist) > 0:
            finalClustersAbove =clustersAbove[min(clustersAboveDist)]
        else: # no above clusters
            aboveMin = np.inf
            for eachRange in levelList:
                for val in eachRange:
                    aboveDist = val - close
    
                    if (aboveDist > 0) and (aboveDist < aboveMin):
                        finalAboveLevel = val
                        aboveMin = aboveDist

        if len(clustersBelowDist) > 0:
            finalClustersBelow = clustersBelow[min(clustersBelowDist)]
        else:  # no below clusters
            belowMin = np.inf
            for eachRange in levelList:
                for val in eachRange:
                    belowDist = close - val

                    if (belowDist > 0) and (belowDist < belowMin):
                        finalBelowLevel = val
                        belowMin = belowDist


    # print(finalAboveLevel,finalBelowLevel,finalClustersAbove, finalClustersBelow)

    clusterColor = 'navy'
    clusterWidth = 3
    clusterDash = 'dash'
    levelColor = 'blue'
    levelWidth = 2
    levelDash = 'dash'

    traces = []

    if len(finalClustersAbove) > 0:

        thisLine = Scatter(name='Resistance Cluster', x=[startX, endX], y=[np.mean(finalClustersAbove), np.mean(finalClustersAbove)],
                           mode='lines',
                           opacity=0.7,
                           line=dict(color=clusterColor,
                                     width=clusterWidth,  # newClusterDict[eachMain]*2,
                                     dash=clusterDash,
                                     ),
                           # marker=dict(symbol='circle'),
                           hoverinfo='y',  # 'none'
                           # legendgroup=name,
                           showlegend=True,
                           # textposition='middle right',
                           # textfont=dict(color=color, family='Gravitas One'),
                           # text=['', '  %.1f' % (retracementPercentages[idx] * 100) + '%']
                           )
        traces.append(thisLine)


    if len(finalClustersBelow) > 0:

        thisLine = Scatter(name='Support Cluster', x=[startX, endX], y=[np.mean(finalClustersBelow), np.mean(finalClustersBelow)],
                           mode='lines',
                           opacity=0.7,
                           line=dict(color=clusterColor,
                                     width=clusterWidth,  # newClusterDict[eachMain]*2,
                                     dash=clusterDash,
                                     ),
                           # marker=dict(symbol='circle'),
                           hoverinfo='y',  # 'none'
                           # legendgroup=name,
                           showlegend=True,
                           # textposition='middle right',
                           # textfont=dict(color=color, family='Gravitas One'),
                           # text=['', '  %.1f' % (retracementPercentages[idx] * 100) + '%']
                           )
        traces.append(thisLine)

    if finalAboveLevel > 0:

        thisLine = Scatter(name='Resistance Rtcmt', x=[startX, endX], y=[np.mean(finalAboveLevel), np.mean(finalAboveLevel)],
                           mode='lines',
                           opacity=0.7,
                           line=dict(color=levelColor,
                                     width=levelWidth,  # newClusterDict[eachMain]*2,
                                     dash=levelDash,
                                     ),
                           # marker=dict(symbol='circle'),
                           hoverinfo='y',  # 'none'
                           # legendgroup=name,
                           showlegend=True,
                           # textposition='middle right',
                           # textfont=dict(color=color, family='Gravitas One'),
                           # text=['', '  %.1f' % (retracementPercentages[idx] * 100) + '%']
                           )
        traces.append(thisLine)

    if finalBelowLevel > 0:

        thisLine = Scatter(name='Support Rtcmt', x=[startX, endX], y=[np.mean(finalBelowLevel), np.mean(finalBelowLevel)],
                           mode='lines',
                           opacity=0.7,
                           line=dict(color=levelColor,
                                     width=levelWidth,  # newClusterDict[eachMain]*2,
                                     dash=levelDash,
                                     ),
                           # marker=dict(symbol='circle'),
                           hoverinfo='y',  # 'none'
                           # legendgroup=name,
                           showlegend=True,
                           # textposition='middle right',
                           # textfont=dict(color=color, family='Gravitas One'),
                           # text=['', '  %.1f' % (retracementPercentages[idx] * 100) + '%']
                           )
        traces.append(thisLine)

    return traces

def clusterAlgo(levelList, minClusters=0,):
    # 1. Method: find min spacing
    # 2. start from biggest range/spacing one.
    # 3. check against the rest.
    # 4. every one that has, take it out. From checker, and checkee. Add to cluster list.
    # 5. keep going until 2nd last guy
    # 6. increase spacing and repeat 2-4 until min clusters met

    rangeList = []
    rangeDict = {}

    # region:find min spacing
    for each in levelList:
        ranger = max(each) - min(each)
        rangeList.append(ranger)
        rangeDict[ranger] = each

    minList = rangeDict[min(rangeList)]

    minSpacing = 1

    for i in range(len(minList) - 1):
        spacing = abs(minList[i] - minList[i + 1]) / 4.0
        if spacing < minSpacing: minSpacing = spacing
    # endregion

    rangeList.sort()
    rangeList.reverse()

    clusterDict = {}

    # main body of this function
    while len(clusterDict) < minClusters:
        print(minSpacing, len(clusterDict))
        for a in range(len(rangeList) - 1):
            mainList = rangeDict[rangeList[a]]
            mainList.sort()

            for mainEl in mainList:

                for b in range(a + 1, len(rangeList)):
                    subList = rangeDict[rangeList[b]]
                    toRemove = []
                    for subEl in subList:
                        diff = abs(mainEl - subEl)

                        if diff < minSpacing:
                            if mainEl in clusterDict:
                                clusterDict[mainEl].append(subEl)
                            else:
                                clusterDict[mainEl] = [subEl]
                                toRemove.append(subEl)

                    for removeEl in toRemove:
                        subList.remove(removeEl)

        keys = list(clusterDict.keys())
        keys.sort()
        keys.reverse()

        # if clusters are still too close, consolidate
        for i in range(len(keys) - 1):
            dist = abs(keys[i] - keys[i + 1])

            if dist <= minSpacing:
                clusterDict[keys[i + 1]].append(keys[i])
                for el in clusterDict[keys[i]]:
                    clusterDict[keys[i + 1]].append(el)
                del clusterDict[keys[i]]

        if minSpacing > max(rangeList):
            print('COULD NOT FIND PRICE CLUSTERS')
            return
        minSpacing *= 1.5  # increase minSpacing to increase num of clusters when not enough clusters

    newClusterDict = {}

    for eachKey in clusterDict.keys():
        this = clusterDict[eachKey]

        avg = (eachKey + sum(this)) / (len(this) + 1)

        newClusterDict[avg] = len(this) + 1

    print(newClusterDict)
    exit()

    return newClusterDict

def getClusters(levelList, startX, endX, minClusters=4,):

    newClusterDict = clusterAlgo(levelList, minClusters,)


    name = 'Cluster Buster'
    color = 'crimson'
    width = 4
    dash = 'solid'

    traces = []
    for eachMain in newClusterDict.keys():
        thisLine = Scatter(name=name, x=[startX,endX], y=[eachMain, eachMain],
                           mode='lines',
                           opacity=0.7,
                           line=dict(color=color,
                                     width=width, #newClusterDict[eachMain]*2,
                                     dash=dash
                                     ),
                           # marker=dict(symbol='circle'),
                           hoverinfo='none',  # 'none'
                           legendgroup=name,
                           showlegend=True,
                           # textposition='middle right',
                           # textfont=dict(color=color, family='Gravitas One'),
                           # text=['', '  %.1f' % (retracementPercentages[idx] * 100) + '%']
                           )
        traces.append(thisLine)

    return traces

def getHps(y):

    hpMask = (
        (y.shift(-1).high <= y.high)
        & (y.shift(1).high <= y.high)
    )
    return y[hpMask]

def getLps(y):

    lpMask = (
        (y.shift(-1).low >= y.low)
        & (y.shift(1).low >= y.low)
    )
    return y[lpMask]

def signalBots(df, bigBots):
    df['midpoint'] = (df.high + df.low) / 2

    mask4 = (df.high > df.shift(1).high) & (df.close < df.shift(1).close) & (df.close < df.open) & (
                df.close < df.midpoint)
    mask3 = (df.high > df.shift(1).high) & (df.close < df.shift(1).close) & (df.close < df.midpoint)
    mask2 = (df.high > df.shift(1).high) & (df.close < df.shift(1).close) & (df.close < df.open)
    mask1 = (df.high > df.shift(1).high) & (df.close < df.shift(1).close)

    level4 = bigBots.merge(df[mask4], on='date')
    level3 = bigBots.merge(df[mask3], on='date')
    level2 = bigBots.merge(df[mask2], on='date')
    level1 = bigBots.merge(df[mask1], on='date')

    allLevels = [level1, level2, level3, level4]

    traces = []
    color = 'darkred'

    for i in range(len(allLevels)):
        # print(len(allLevels[i]))

        thisTrace = Scatter(
            x=allLevels[i].date,
            y=allLevels[i].point,
            name='Lvl %i Potential Signal Bottoms' % (i + 1),
            mode='markers',
            marker=dict(color=color, size=2 + (3 * i)),
            hoverinfo='none',
            legendgroup='Signal Bottoms',
            showlegend=True,
        )
        traces.append(thisTrace)

    return traces

def signalTops(df, bigTops):


    df['midpoint'] = (df.high + df.low)/2

    mask4 = (df.high > df.shift(1).high) & (df.close< df.shift(1).close) & (df.close<df.open) & (df.close < df.midpoint)
    mask3 = (df.high > df.shift(1).high) & (df.close < df.shift(1).close) & (df.close < df.midpoint)
    mask2 = (df.high > df.shift(1).high) & (df.close < df.shift(1).close) & (df.close < df.open)
    mask1 = (df.high > df.shift(1).high) & (df.close < df.shift(1).close)

    level4 = bigTops.merge(df[mask4],on='date')
    level3 = bigTops.merge(df[mask3],on='date')
    level2 = bigTops.merge(df[mask2],on='date')
    level1 = bigTops.merge(df[mask1],on='date')

    allLevels = [level1,level2,level3,level4]
    traces = []
    color = 'red'

    for i in range(len(allLevels)):

        # print(len(allLevels[i]))

        thisTrace = Scatter(
            x = allLevels[i].date,
            y = allLevels[i].point,
            name='Lvl %i Potential Signal Tops' %(i+1),
            mode='markers',
            marker=dict(color=color,size=2+(3*i)),
            hoverinfo='none',
            legendgroup='Signal Tops',
            showlegend=True,
        )
        traces.append(thisTrace)

    return traces

def plotTimeRets(firstDate,lastDate,maxHeight,minHeight):

    retPercentages = [0.25, 0.5, 0.75, 1]
    timeRange = lastDate-firstDate
    newDates =[]

    for percent in retPercentages:
        newDates.append(datetime.date(percent*timeRange + lastDate))

    traces = []
    color = 'brown'
    width = 1
    dash = 'dash'  # 'dot', 'dashdot'


    for idx in range(len(newDates)):
        showLegend=False
        if idx == 0: showLegend=True

        thisLine= Scatter(name='Time Levels', x=[newDates[idx],newDates[idx]], y=[minHeight,maxHeight],
                      mode='lines+text',
                      line=dict(color=color,
                                width=width,
                                dash=dash
                                ),
                      # marker=dict(symbol='circle'),
                      hoverinfo='x', #'none'
                          hovertext='%.2f'%(retPercentages[idx] * 100) + '%',

                     legendgroup='Time',
                          showlegend=showLegend,
                          textposition='top left',
                          textfont=dict(color=color, family='Gravitas One'),
                          text=['','  %i'%(retPercentages[idx] * 100) + '%']
                          )


        traces.append(thisLine)

    return traces

    exit()

def plotGannAngles(x0_date,x0_idx,xLast_date,xLast_idx,y0, trendUp = False, ratio=1,scale=1,name = '1x1',color='navy'):



    m = ratio * scale

    if not trendUp: m *= -1
    c = y0 - (m*x0_idx)
    yLast = (xLast_idx*m) + c


    line = Scatter(name=name + ' Gann',
                   x=[x0_date,xLast_date], y=[y0,yLast],
                   mode='lines+text',
                   line=dict(
                       color=color,
                        width=2,
                        dash='solid',

                             ),
                   hoverinfo='none',
                   showlegend=True,
                   opacity=0.7,
                   text=['','   '+name],
                   textfont=dict(
                       color = color,
                       family='Gravitas One',
                   ),
                   textposition='middle right',
                   )

    return line

def retracementLines(firstPoint,lastPoint,x, name='Retracement',color='#000080',previousLowProjLevels=[]):
    # lastPoint = lastPoint.values[0]
    # firstPoint = firstPoint.valuesalues[0]

    temp = lastPoint
    lastPoint = firstPoint
    firstPoint = temp

    trendRange = lastPoint - firstPoint

    levels = []
    lowProj = []
    lowProjLevels =[]

    retracementPercentages = [.25,.33,.382,.50,.618,.66,.75]

    if trendRange < 0: #uptrend
        retracementPercentages += [.10, .20, .30, .40, .60, .70, .80, .90, .125, .75, .875]

    else:
        # retracementPercentages += []
        lowProj = [1.5,
                   2, 2.5, 3, 4, 5]


    for percent in retracementPercentages:
        levels.append(percent * trendRange + firstPoint)

    for multiple in lowProj:
        val = multiple * firstPoint

        if not (val in previousLowProjLevels):
            # levels.append(val)
            lowProjLevels.append(val)



    traces = []
    # annot = []

    # color = '#000080'
    width = 1
    dash = 'longdashdot' #'dot', 'dashdot'


    firstLine = Scatter(name=name, x=x, y=[levels[0],levels[0]],
                      mode='lines+text',
                      line=dict(color=color,
                                width=width,
                                dash=dash
                                ),
                      # marker=dict(symbol='circle'),
                      hoverinfo='none',
                     legendgroup=name,
                        textposition='middle right',
                        textfont=dict(color=color, family='Gravitas One'),
                        text=['', '   %.1f' % (retracementPercentages[0] * 100) + '%']

                      )
    # annot.append(dict(x=x[-1],y=levels[0],text=str(retracementPercentages[0]),showarrow=False,ax=0,ay=-30,font=dict(color=color)))
    traces.append(firstLine)

    for idx in range(1,len(retracementPercentages)):
        thisLine= Scatter(name=name, x=x, y=[levels[idx],levels[idx]],
                      mode='lines+text',
                      line=dict(color=color,
                                width=width,
                                dash=dash
                                ),
                      # marker=dict(symbol='circle'),
                      hoverinfo='none', #'none'
                     legendgroup=name,
                          showlegend=False,
                          textposition='middle right',
                          textfont=dict(color=color, family='Gravitas One'),
                          text=['','  %.1f'%(retracementPercentages[idx] * 100) + '%']
                          )

        # annot.append(dict(x=x[-1], y=levels[idx], text=str(retracementPercentages[idx]), showarrow=False, ax=0, ay=-30,font=dict(color=color)))
        traces.append(thisLine)

    for idx in range(len(lowProjLevels)):
        thisLine= Scatter(name=name, x=x, y=[lowProjLevels[idx],lowProjLevels[idx]],
                      mode='lines+text',
                      line=dict(color=color,
                                width=width,
                                dash=dash
                                ),
                      # marker=dict(symbol='circle'),
                      hoverinfo='none', #'none'
                     legendgroup=name,
                          showlegend=False,
                          textposition='middle right',
                          textfont=dict(color=color, family='Gravitas One'),
                          text=['','  %.1f'%(lowProj[idx]) + 'x']
                          )

        # annot.append(dict(x=x[-1], y=levels[idx], text=str(retracementPercentages[idx]), showarrow=False, ax=0, ay=-30,font=dict(color=color)))
        traces.append(thisLine)

    return traces, levels, lowProjLevels

def check3Points(counter, topsAndBottoms, trendUp, topOrBottom):
    if not (
            (topsAndBottoms.iloc[counter].trendUp == trendUp) and
            (topsAndBottoms.iloc[counter - 1].trendUp == trendUp) and
            (topsAndBottoms.iloc[counter - 2].trendUp == trendUp) and
            (topsAndBottoms.iloc[counter].top == topOrBottom) and
            (topsAndBottoms.iloc[counter - 1].top == ~topOrBottom) and
            (topsAndBottoms.iloc[counter - 2].top == topOrBottom)
    ):
        return False, counter - 1

    diff = topsAndBottoms.iloc[counter].date - topsAndBottoms.iloc[counter - 2].date

    return diff.days, counter - 2


def trendProjector(topsAndBottoms, todaysDate, highColor = 'orange', lowColor = 'violet'):
    # 1. have a 3d list of tops, bottoms and projections
    # 2. have 2 charts. 1 to project tops, another one to project bottoms.

    topsAndBottoms = topsAndBottoms.reset_index(drop=True)

    HH_days = set()
    HL_days = set()
    LL_days = set()
    LH_days = set()

    HH_projs = {}
    HL_projs = {}
    LL_projs = {}
    LH_projs = {}

    HHstart,HHend,HHdiff = [],[],[]
    HLstart, HLend, HLdiff = [], [], []
    LLstart, LLend, LLdiff = [], [], []
    LHstart, LHend, LHdiff = [], [], []

    totalLength = len(topsAndBottoms)

    barWidth = 1

    latestDate = todaysDate

    for index,row in topsAndBottoms.iterrows():
        if row.top:
            #calculate projections
            projectionList = []
            for projDays in HH_days:
                projection = row.date + timedelta(days=projDays)
                if projection >= todaysDate:
                    projectionList.append(projection)
            if projectionList != []: HH_projs[row.date] = projectionList

            projectionList = []
            for projDays in HL_days:
                projection = row.date + timedelta(days=projDays)
                if projection >= todaysDate:
                    projectionList.append(projection)
            if projectionList != []: HL_projs[row.date] = projectionList

            #gather new data
            if (index+1) < totalLength:
                diff = topsAndBottoms.iloc[index + 1].date - row.date
                HL_days.add(diff.days)

                HLstart.append(row.date)
                HLend.append(topsAndBottoms.iloc[index + 1].date)
                HLdiff.append(diff.days)

            if (index+2) < totalLength:
                diff = topsAndBottoms.iloc[index+2].date - row.date
                HH_days.add(diff.days)

                HHstart.append(row.date)
                HHend.append(topsAndBottoms.iloc[index + 2].date)
                HHdiff.append(diff.days)


        else: # bottoms
            # calculate projections
            projectionList = []
            for projDays in LL_days:
                projection = row.date + timedelta(days=projDays)
                if projection >= todaysDate:
                    projectionList.append(projection)
            if projectionList!=[] : LL_projs[row.date] = projectionList

            projectionList = []
            for projDays in LH_days:
                projection = row.date + timedelta(days=projDays)
                if projection >= todaysDate:
                    projectionList.append(projection)
            if projectionList != []: LH_projs[row.date] = projectionList

            # gather new data
            if (index + 1) < totalLength:
                diff = topsAndBottoms.iloc[index + 1].date - row.date
                LH_days.add(diff.days)

                LHstart.append(row.date)
                LHend.append(topsAndBottoms.iloc[index + 1].date)
                LHdiff.append(diff.days)

            if (index + 2) < totalLength:
                diff = topsAndBottoms.iloc[index + 2].date - row.date
                LL_days.add(diff.days)

                LLstart.append(row.date)
                LLend.append(topsAndBottoms.iloc[index + 2].date)
                LLdiff.append(diff.days)


    # Projection of next High

    Hproj_bars = []
    ganttList = []

    for eachProj in HH_projs:
        eachDates = HH_projs[eachProj]

        #Get Latest Projection Date
        latestDate = max([latestDate, max(eachDates)])

        length = len(eachDates)

        thisBar = Bar(x=eachDates, y=[1]*length,
                      # width=[barWidth] * length,
                      # xbins=dict(start=np.min(HH_bars), size=size, end=np.max(HH_bars)),
                      # hoverinfo='none',
                      name='HH Projection from '+eachProj.strftime("%y-%m-%d"),
                      dx=1,dy=1,
                      yaxis='y2',
                      legendgroup='Proj of next highs',
                      showlegend=False,
                      opacity=0.4,
                      marker=dict(color=highColor),
                      hoverinfo='x',
                      )

        Hproj_bars.append(thisBar)

        for date in eachDates:
            thisGantt = dict(Task=eachProj,Start=date,Finish=date+timedelta(days=7))
            ganttList.append(thisGantt)

    for eachProj in LH_projs:
        eachDates = LH_projs[eachProj]

        # Get Latest Projection Date
        latestDate = max([latestDate, max(eachDates)])

        length = len(eachDates)

        thisBar = Bar(x=eachDates, y=[1]*length,
                      # width=[barWidth] * length,
                      # xbins=dict(start=np.min(HH_bars), size=size, end=np.max(HH_bars)),
                      # hoverinfo='none',
                      name='LH Projection from '+eachProj.strftime("%y-%m-%d"),
                      dx=1,dy=1,
                      yaxis='y2',
                      legendgroup='Proj of next highs',
                      showlegend=False,
                      opacity=0.4,
                      marker=dict(color=highColor),
                      hoverinfo='x',
                      )

        Hproj_bars.append(thisBar)

        for date in eachDates:
            thisGantt = dict(Task=eachProj,Start=date,Finish=date+timedelta(days=7))
            ganttList.append(thisGantt)



    layout = Layout(
        barmode='stack',title='Projection of next Top'
    )

    fig = Figure(data=Hproj_bars, layout=layout)

    # gantt = ff.create_gantt(ganttList,showgrid_x=True,showgrid_y=True,height=900,width=1200)



    # Projection of next Low

    Lproj_bars = []
    ganttListLow = []

    for eachProj in LL_projs:
        eachDates = LL_projs[eachProj]

        # Get Latest Projection Date
        latestDate = max([latestDate, max(eachDates)])

        length = len(eachDates)

        thisBar = Bar(x=eachDates, y=[1]*length,
                      # width=[barWidth] * length,
                      # xbins=dict(start=np.min(HH_bars), size=size, end=np.max(HH_bars)),
                      # hoverinfo='none',
                      name='LL Projection from '+eachProj.strftime("%y-%m-%d"),
                      dx=1,dy=1,
                      yaxis='y2',
                      legendgroup='Proj of next lows',
                      showlegend=False,
                      opacity=0.4,
                      marker=dict(color=lowColor),
                      hoverinfo='x',
                      )

        Lproj_bars.append(thisBar)

        for date in eachDates:
            thisGantt = dict(Task=eachProj,Start=date,Finish=date+timedelta(days=7))
            ganttListLow.append(thisGantt)

    for eachProj in HL_projs:
        eachDates = HL_projs[eachProj]

        # Get Latest Projection Date
        latestDate = max([latestDate, max(eachDates)])

        length = len(eachDates)

        thisBar = Bar(x=eachDates, y=[1]*length,
                      # width=[barWidth]*length,
                      # xbins=dict(start=np.min(HH_bars), size=size, end=np.max(HH_bars)),
                      # hoverinfo='none',
                      name='HL Projection from '+eachProj.strftime("%y-%m-%d"),
                      dx=1,dy=1,
                      yaxis='y2',
                      legendgroup='Proj of next lows',
                      showlegend=False,
                      opacity=0.4,
                      marker=dict(color=lowColor),
                      hoverinfo='x',
                      )

        Lproj_bars.append(thisBar)

        for date in eachDates:
            thisGantt = dict(Task=eachProj,Start=date,Finish=date+timedelta(days=7))
            ganttListLow.append(thisGantt)



    layout = Layout(
        barmode='stack',title='Projection of next Bottom'
    )

    figLow = Figure(data=Lproj_bars, layout=layout)

    ganttLow = ff.create_gantt(ganttListLow,showgrid_x=True,showgrid_y=True,height=900,width=1200)

    # tables for diagnostics
    HH_table = Figure(data=[Table(header=dict(values=['Start','End','No. of days']),
                     cells=dict(values=[HHstart,HHend,HHdiff]),name='HH interval data')], layout=Layout(title='HH interval data'))
    HL_table = Figure(data=[Table(header=dict(values=['Start','End','No. of days']),
                     cells=dict(values=[HLstart,HLend,HLdiff]),name='HL interval data')], layout=Layout(title='HL interval data'))
    LL_table = Figure(data=[Table(header=dict(values=['Start','End','No. of days']),
                     cells=dict(values=[LLstart,LLend,LLdiff]),name='LL interval data')], layout=Layout(title='LL interval data'))
    LH_table = Figure(data=[Table(header=dict(values=['Start','End','No. of days']),
                     cells=dict(values=[LHstart,LHend,LHdiff]),name='LH interval data')], layout=Layout(title='LH interval data'))



    # figTable = plotly.tools.make_subplots(rows=2, cols=2, subplot_titles=('HH interval data',
    #                                                                       'LL interval data',
    #                                                                       'HL interval data',
    #                                                                       'LH interval data',))
    # figTable.append_trace(HH_table, 1, 1)
    # figTable.append_trace(LL_table, 1, 2)
    # figTable.append_trace(HL_table, 2, 1)
    # figTable.append_trace(LH_table, 2, 2)




    return Hproj_bars,Lproj_bars, fig, figLow, latestDate
        # plotly.offline.plot(figure_or_data=fig,
        #                        show_link=False,
        #                        output_type='div',
        #                        include_plotlyjs=False,
        #                        # filename='minorHLData.html',
        #                        auto_open=False,
        #                        config={'displaylogo': False,
        #                                'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'zoomIn2d',
        #                                                           'zoomOut2d',
        #                                                           'resetScale2d', 'hoverCompareCartesian', 'lasso2d'],
        #                                'displayModeBar': False
        #                                }),\
        #    plotly.offline.plot(figure_or_data=figLow,
        #                        show_link=False,
        #                        output_type='div',
        #                        include_plotlyjs=False,
        #                        # filename='minorHLData.html',
        #                        auto_open=False,
        #                        config={'displaylogo': False,
        #                                'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'zoomIn2d',
        #                                                           'zoomOut2d',
        #                                                           'resetScale2d', 'hoverCompareCartesian', 'lasso2d'],
        #                                'displayModeBar': False
        #                                }),\
        #    plotly.offline.plot(figure_or_data=HH_table,
        #                        show_link=False,
        #                        output_type='div',
        #                        include_plotlyjs=False,
        #                        # filename='minorHLData.html',
        #                        auto_open=False,
        #                        config={'displaylogo': False,
        #                                'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'zoomIn2d',
        #                                                           'zoomOut2d',
        #                                                           'resetScale2d', 'hoverCompareCartesian', 'lasso2d'],
        #                                'displayModeBar': False
        #                                }),\
        #    plotly.offline.plot(figure_or_data=LH_table,
        #                        show_link=False,
        #                        output_type='div',
        #                        include_plotlyjs=False,
        #                        # filename='minorHLData.html',
        #                        auto_open=False,
        #                        config={'displaylogo': False,
        #                                'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'zoomIn2d',
        #                                                           'zoomOut2d',
        #                                                           'resetScale2d', 'hoverCompareCartesian', 'lasso2d'],
        #                                'displayModeBar': False
        #                                }),\
        #    plotly.offline.plot(figure_or_data=LL_table,
        #                        show_link=False,
        #                        output_type='div',
        #                        include_plotlyjs=False,
        #                        # filename='minorHLData.html',
        #                        auto_open=False,
        #                        config={'displaylogo': False,
        #                                'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'zoomIn2d',
        #                                                           'zoomOut2d',
        #                                                           'resetScale2d', 'hoverCompareCartesian', 'lasso2d'],
        #                                'displayModeBar': False
        #                                }),\
        #    plotly.offline.plot(figure_or_data=HL_table,
        #                        show_link=False,
        #                        output_type='div',
        #                        include_plotlyjs=False,
        #                        # filename='minorHLData.html',
        #                        auto_open=False,
        #                        config={'displaylogo': False,
        #                                'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'zoomIn2d',
        #                                                           'zoomOut2d',
        #                                                           'resetScale2d', 'hoverCompareCartesian', 'lasso2d'],
        #                                'displayModeBar': False
        #                                })

# plotly.offline.plot(figure_or_data=gantt,
           #                     show_link=False,
           #                     output_type='div',
           #                     include_plotlyjs=False,
           #                     # filename='minorHLData.html',
           #                     auto_open=False,
           #                     config={'displaylogo': False,
           #                             'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'zoomIn2d',
           #                                                        'zoomOut2d',
           #                                                        'resetScale2d', 'hoverCompareCartesian', 'lasso2d'],
           #                             'displayModeBar': False
           #                             }),\
    # plotly.offline.plot(figure_or_data=ganttLow,
    #                     show_link=False,
    #                     output_type='div',
    #                     include_plotlyjs=False,
    #                     # filename='minorHLData.html',
    #                     auto_open=False,
    #                     config={'displaylogo': False,
    #                             'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'zoomIn2d',
    #                                                        'zoomOut2d',
    #                                                        'resetScale2d', 'hoverCompareCartesian', 'lasso2d'],
    #                             'displayModeBar': False
    #                             }), \
 # \

def trendFinder(stuff,barHeight=0.03,upColor = 'green',downColor='red'):

    if 'topsAndBottoms' in stuff.keys():
        topsAndBottoms = stuff['topsAndBottoms']
    else:
        topsAndBottoms = stuff

    # barHeight = 0.03
    # barBias =

    trendUp = pd.DataFrame()
    trendUp['date'] = topsAndBottoms.date
    trendUp['point'] = topsAndBottoms.point
    # topsAndBottoms['trendUp'] = False

    # True if: is a top and breaks previous top,
    # True if: is a top and does not break previous top but previous trend is True
    # True if: is a bottom and does not break previous bottom and previous trend is True

    # topsAndBottoms['trendUp'] = (topsAndBottoms.top) & (topsAndBottoms.point > topsAndBottoms.shift(2).point)
    #
    # topsAndBottoms['trendUp'] = (topsAndBottoms.top &
    #                              (
    #                                  (topsAndBottoms.point > topsAndBottoms.shift(2).point)|
    #                                  topsAndBottoms.shift(1).trendUp
    #                              ))| (topsAndBottoms.bottom &
    #                                   (topsAndBottoms.point > topsAndBottoms.shift(2).point) &
    #                                   topsAndBottoms.shift(1).trendUp
    #                                   )

    upTrendList=[]

    topsAndBottoms.reset_index(drop=True,inplace=True)
    for index,row in topsAndBottoms.iterrows():
        if index < 2:
            upTrendList.append(False)
            continue
        if row.top:
            if row.point > topsAndBottoms.iloc[index-2].point: #broken previous top
                upTrendList.append(True)
            else:
                #did not break previous top
                upTrendList.append(upTrendList[index-1])
        else: #if row.bottom
            if row.point >= topsAndBottoms.iloc[index-2].point:
                upTrendList.append(upTrendList[index - 1])
            else:
                upTrendList.append(False) #broke previous bottom
    # map(lambda x: x * 2, l)
    # upTrendList =
    trendUp.reset_index(drop=True, inplace=True)


    trendUp['trendUp'] = [x * barHeight * 2 - barHeight for x in upTrendList]
    # trendUp.reset_index(drop=True, inplace=True)

    trendUp.ix[0, 'trendUp'] = 0
    trendUp.ix[1, 'trendUp'] = 0

    ups = trendUp[trendUp.trendUp > 0]
    downs = trendUp[trendUp.trendUp < 0]
    # print(ups)
    # print(downs)

    upsTrace = Bar(name='Trend Up', x=ups.date, y=ups.trendUp, marker=dict(color=upColor),
                   text='up', textposition='auto', opacity=0.7,
                   # showlegend=False,
                   hoverinfo="text")
    downsTrace = Bar(name='Trend Down', x=downs.date, y=downs.trendUp, marker=dict(color=downColor),
                     text='down', textposition='auto', opacity=0.7,
                     # showlegend=False,
                     hoverinfo='text')

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

    stringTrend = 'Current Trend is: '
    if upTrendList[-1]:
        stringTrend += 'UP'
    else:
        stringTrend += 'DOWN'

    lastDate = None
    firstDate = None

    currentTrend = trendUp.iloc[-1].trendUp
    prevTrend = False
    for i in range(len(trendUp)-1,-1,-1):

        if (trendUp.iloc[i].trendUp != currentTrend) and not prevTrend:
            if trendUp.iloc[i].trendUp > 0: # trending up
                topPoints = []
                topDict = {}
                for x in range(i,-1,-2):
                    if trendUp.iloc[x].trendUp < 0: break
                    topPoints.append(trendUp.iloc[x].point)
                    topDict[trendUp.iloc[x].point] = trendUp.iloc[x].date

                lastDate = topDict[max(topPoints)]


            elif trendUp.iloc[i].trendUp < 0: # trending down
                botPoints = []
                botDict = {}
                for x in range(i,-1,-2):
                    if trendUp.iloc[x].trendUp > 0: break
                    botPoints.append(trendUp.iloc[x].point)
                    botDict[trendUp.iloc[x].point] = trendUp.iloc[x].date
                lastDate = botDict[min(botPoints)]


            prevTrend = True
            continue

        if (trendUp.iloc[i].trendUp == currentTrend) and prevTrend:

            if trendUp.iloc[i].trendUp > 0:  # trending up
                topPoints = []
                topDict = {}
                for x in range(i, -1, -2):
                    if trendUp.iloc[x].trendUp < 0: break
                    topPoints.append(trendUp.iloc[x].point)
                    topDict[trendUp.iloc[x].point] = trendUp.iloc[x].date

                firstDate = topDict[max(topPoints)]


            elif trendUp.iloc[i].trendUp < 0:  # trending down
                botPoints = []
                botDict = {}
                for x in range(i, -1, -2):
                    if trendUp.iloc[x].trendUp > 0: break
                    botPoints.append(trendUp.iloc[x].point)
                    botDict[trendUp.iloc[x].point] = trendUp.iloc[x].date
                firstDate = botDict[min(botPoints)]



            # firstDate = trendUp.iloc[i].date
            break

    # print('first:', firstDate)
    # print('last:', lastDate)

    topsAndBottoms['trendUp'] = trendUp.trendUp > 0
    topsAndBottoms['bigTop'] = False
    topsAndBottoms['bigBot'] = False

    bigTops = []
    bigBots = []
    pointList = []
    pointDict = {}
    currTrendUp = False

    for idx,row in topsAndBottoms.iterrows():
        if row.trendUp!=currTrendUp:
            if currTrendUp: #looking for tops
                bigTops.append(pointDict[max(pointList)])
            else: # looking for bottoms
                bigBots.append(pointDict[min(pointList)])

            #reset
            pointList=[]
            pointDict={}
            currTrendUp = row.trendUp

        if row.trendUp:
            if row.top:
                pointList.append(row.point)
                pointDict[row.point]=idx

        else:
            if row.bottom:
                pointList.append(row.point)
                pointDict[row.point] = idx

    for i in bigTops:
        topsAndBottoms.ix[i,'bigTop'] = True

    for i in bigBots:
        topsAndBottoms.ix[i,'bigBot'] = True

    # for idx,row

    return upsTrace, downsTrace, topsAndBottoms, stringTrend, firstDate, lastDate, currentTrend


def plotTopBotHist(stuff):
    HH_bars = stuff['HH_bars']
    HH_barsMean = np.mean(HH_bars)
    # HH_barsMode = sp.stats.mode(HH_bars).mode[0]
    HH_title = 'Average TOP to TOP duration: %.2f' % (HH_barsMean) + ' bars'
    HH_bars = HH_bars.value_counts()
    HH_barsTrace = Bar(x=HH_bars.index, y=HH_bars.values,
                       # xbins=dict(start=np.min(HH_bars), size=size, end=np.max(HH_bars)),
                       showlegend=False,
                       hoverinfo='none',
                       )

    LL_bars = stuff['LL_bars']
    LL_barsMean = np.mean(LL_bars)
    # LL_barsMode = sp.stats.mode(LL_bars).mode[0]
    LL_title = 'Average BOTTOM to BOTTOM duration: %.2f' % (LL_barsMean) + ' bars'
    LL_bars = LL_bars.value_counts()
    LL_barsTrace = Bar(x=LL_bars.index, y=LL_bars.values,
                       # xbins=dict(start=np.min(HH_bars), size=size, end=np.max(HH_bars)),
                       showlegend=False,
                       hoverinfo='none',
                       )

    HL_bars = stuff['HL_bars']
    HL_barsMean = np.mean(HL_bars)
    # HL_barsMode = sp.stats.mode(HL_bars).mode[0]
    HL_title = 'Average TOP to BOTTOM duration: %.2f' % (HL_barsMean) + ' bars'
    HL_bars = HL_bars.value_counts()
    HL_barsTrace = Bar(x=HL_bars.index, y=HL_bars.values,
                       # xbins=dict(start=np.min(HH_bars), size=size, end=np.max(HH_bars)),
                       showlegend=False,
                       hoverinfo='none',
                       )

    LH_bars = stuff['LH_bars']
    LH_barsMean = np.mean(LH_bars)
    # LH_barsMode = sp.stats.mode(LH_bars).mode[0]
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
                         xaxis={'title': 'Duration in Bars', 'dtick': 1},
                         yaxis={'title': 'Frequency count', 'dtick': 1},
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
                                       'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'zoomIn2d',
                                                                  'zoomOut2d',
                                                                  'resetScale2d', 'hoverCompareCartesian', 'lasso2d'],
                                       'displayModeBar': False
                                       }),


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

    for index, row in months.iterrows():
        thisMonth = (df.date.dt.month == row.month) & (df.date.dt.year == row.year)
        monthlyData = df[thisMonth]

        maxPrice = monthlyData.loc[monthlyData['high'].idxmax()]
        minPrice = monthlyData.loc[monthlyData['low'].idxmin()]

        earliest = monthlyData.loc[monthlyData['date'].idxmin()]
        latest = monthlyData.loc[monthlyData['date'].idxmax()]

        months.at[index, 'high'] = maxPrice.high
        months.at[index, 'low'] = minPrice.low
        months.at[index, 'open'] = earliest.open
        months.at[index, 'close'] = latest.close

        if maxPrice.date < minPrice.date:
            months.at[index, 'lowFirst'] = False

    months['date'] = months.apply(lambda row: datetime(
        row['year'], row['month'], 1), axis=1)

    return pd.DataFrame(months)


def plotter(figure, filename, htmlList):
    plotly.offline.plot(figure_or_data=figure,
                        show_link=False,
                        output_type='file',
                        include_plotlyjs=False,
                        filename=filename,
                        auto_open=True,
                        config={'displaylogo': False,
                                'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'zoomIn2d', 'zoomOut2d',
                                                           'resetScale2d', 'hoverCompareCartesian', 'lasso2d']})
    line = '<script src="plotly-latest.min.js"></script>'
    with open(filename, 'r+') as htmlFile:
        contents = htmlFile.read()
        htmlFile.seek(0, 0)
        htmlFile.write(line.rstrip('\r\n') + '\n' + contents)

        for eachHtml in htmlList:
            htmlFile.write('\n' + eachHtml)


def plotTrendlines(trendLine, stuff, name, color, width, dash=None):
    line = Scatter(name=name + 'Trend', x=trendLine.date, y=trendLine.point,
                   mode='lines',
                   line=dict(color=color,
                             width=width,
                             dash=dash
                             ),
                   hoverinfo='none',
                   showlegend=True,
                   )

    tops = Scatter(name=name + ' Tops', x=stuff['tops'].date, y=stuff['tops'].point,
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

    bottoms = Scatter(name=name + ' Bottoms', x=stuff['bottoms'].date, y=stuff['bottoms'].point,
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

    return line, tops, bottoms


def getTrendTopsAndBottoms(trendLine, df):
    tops = ((trendLine.point > trendLine.shift(1).point) & (trendLine.point > trendLine.shift(-1).point))
    bottoms = ((trendLine.point < trendLine.shift(1).point) & (trendLine.point < trendLine.shift(-1).point))

    trendLine['top'] = tops
    trendLine['bottom'] = bottoms

    reindexed = trendLine.reset_index(drop=True)

    dateIndexOnly = pd.DataFrame()
    dateIndexOnly['i'] = df.reset_index(drop=True).index
    dateIndexOnly['date'] = df.date
    reindexed = reindexed.merge(dateIndexOnly, on='date')

    topPoints = pd.DataFrame(reindexed[tops])
    # topPoints['i'] = topPoints.i
    bottomPoints = pd.DataFrame(reindexed[bottoms])
    # bottomPoints['i'] = bottomPoints.i
    topAndBottomPoints = pd.DataFrame(reindexed[tops | bottoms])
    # topAndBottomPoints['i'] = topAndBottomPoints.i

    # region: High to High
    HH_time = topPoints.date - topPoints.shift(1).date
    HH_price = topPoints.point - topPoints.shift(1).point
    HH_bars = topPoints.i - topPoints.shift(1).i

    HH_time = HH_time.dropna()
    HH_price = HH_price.dropna()

    HH_bars = HH_bars.dropna()

    # fig = ff.create_distplot([HH_bars], group_labels=['test1'],
    # endregion

    # region: Low to Low
    LL_time = bottomPoints.date - bottomPoints.shift(1).date
    LL_price = bottomPoints.point - bottomPoints.shift(1).point
    LL_bars = bottomPoints.i - bottomPoints.shift(1).i

    LL_time = LL_time.dropna()
    LL_price = LL_price.dropna()
    LL_bars = LL_bars.dropna()

    LL_barsMean = np.mean(LL_bars)
    # LL_barsMode = sp.stats.mode(LL_bars).mode[0]

    LL_barsTrace = Histogram(x=LL_bars, xbins=dict(start=np.min(LL_bars), size=1, end=np.max(LL_bars)))
    LL_barsFig = Figure(data=[LL_barsTrace])
    # endregion

    # region: High to Low and Low to High
    mixed_time = topAndBottomPoints.date - topAndBottomPoints.shift(1).date
    mixed_price = topAndBottomPoints.point - topAndBottomPoints.shift(1).point
    mixed_bars = topAndBottomPoints.i - topAndBottomPoints.shift(1).i

    mixed_time = mixed_time.dropna().reset_index(drop=True)
    mixed_price = mixed_price.dropna().reset_index(drop=True)
    mixed_bars = mixed_bars.dropna().reset_index(drop=True)

    topAndBottomPoints.reset_index(drop=True, inplace=True)

    if topAndBottomPoints.iloc[0].point < topAndBottomPoints.iloc[1].point:  # bottom first
        HL_time = mixed_time[mixed_time.index % 2 == 0]  # even
        LH_time = mixed_time[mixed_time.index % 2 == 1]  # odd

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

    # endregion

    topText = []
    bottomText = []
    topAndBottomPoints['xDiff'] = topAndBottomPoints.i - topAndBottomPoints.shift(1).i
    topAndBottomPoints['yDiff'] = topAndBottomPoints.point - topAndBottomPoints.shift(1).point

    topAndBottomPoints.fillna(value=0,inplace=True)



    for index, row in topAndBottomPoints.iterrows():
        if row.top:
            text = str(row.point) + '<br>'+ str(int(row.xDiff)) + '<br>' + '%.4f'%(row.yDiff)
            topText.append(text)
            # bottomText.append('')
        elif row.bottom:
            # topText.append('')
            text = str(row.point) + '<br>' + str(int(row.xDiff)) + '<br>' + '%.4f' % (row.yDiff)
            bottomText.append(text)
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
                topsAndBottoms=trendLine[tops | bottoms],
                topText=topText,
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


def processOutsideBars(row, trendPoints, DELAY, minorPoints):
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
    dfIgnoreInsideBars['outside'] = (dfIgnoreInsideBars.high > dfIgnoreInsideBars.shift(1).high) & (
                dfIgnoreInsideBars.low < dfIgnoreInsideBars.shift(1).low)

    minorPoints = []
    intermediatePoints = []
    majorPoints = []

    # dfIgnoreInsideBars = pd.DataFrame(df)

    for index, row in dfIgnoreInsideBars.iterrows():
        if index < 1: continue
        if row.outside:

            if index >= 2: intermediatePoints = processOutsideBars(row, intermediatePoints, DELAY=2,
                                                                   minorPoints=minorPoints)
            if index >= 3: majorPoints = processOutsideBars(row, majorPoints, DELAY=3, minorPoints=minorPoints)
            minorPoints = processOutsideBars(row, minorPoints, DELAY=1, minorPoints=minorPoints)
            continue

        # minor points
        result = checkPrevPoints(dfIgnoreInsideBars, index, row, DELAY=1)
        if result != None: minorPoints.append(result)

        # intermediate points
        if index >= 2:
            result = checkPrevPoints(dfIgnoreInsideBars, index, row, DELAY=2)
            if result != None: intermediatePoints.append(result)

        # major points
        if index >= 3:
            result = checkPrevPoints(dfIgnoreInsideBars, index, row, DELAY=3)
            if result != None: majorPoints.append(result)

    trendLine1 = pd.DataFrame(minorPoints, columns=['date', 'point'])
    trendLine2 = pd.DataFrame(intermediatePoints, columns=['date', 'point'])
    trendLine3 = pd.DataFrame(majorPoints, columns=['date', 'point'])

    return trendLine1, trendLine2, trendLine3

