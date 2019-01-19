from trendFunctions import *
from hurstAnalysis2 import *


def main():
    startTime = time.time()

    df = pd.read_excel('EURUSD Weekly Data for Swing Indicator.xlsx')

    # df = pd.read_csv('EURUSD Weekly Data for Swing Indicator.csv')

    # df = pd.read_excel('test1.xlsx')

    # Convert Date Format
    df.columns = ['date', 'close', 'open', 'high', 'low']
    df['date'] = pd.to_datetime(df['date'])

    #only retain data for last 5 years
    # df = df[df.date > (df.iloc[-1].date-timedelta(days=(365.25*2)))]
    # df = df.reset_index(drop=True)



    cutoff = len(df) - 0

    dfTail = df[cutoff:]
    df = df[:cutoff]

    backgroundColor = 'black'

    # endpoint = df.iloc[-1].date + timedelta(days=(
    #     (df.iloc[-1].date - df.iloc[0].date).days)*0.30)

    endpoint = df.iloc[-1].date + timedelta(weeks=24)
    
    xaxisRange = [
        df.iloc[0].date,
        # df.iloc[-1].date,
        endpoint
    ]

    # endpoint = dfTail.iloc[-1].date + timedelta(days=((df.iloc[-1].date - df.iloc[0].date).days) * 0.10)

    # df = groupByMonths(df)

    # region: add lowFirst column if no timing data
    if 'lowFirst' not in df.columns: df['lowFirst'] = df.open < df.close
    # endregion
    lengthDf = len(df)
    df = pd.DataFrame(df.iloc[0:]).reset_index(drop=True)

    # inside bars (fully ignore for trend line calculation)
    # region: INSIDE BARS

    activeDate = []
    activeClose = []
    activeOpen = []
    activeHigh = []
    activeLow = []
    activeLowFirst = []
    activeBarIndex = []

    insideDate = []
    insideClose = []
    insideOpen = []
    insideHigh = []
    insideLow = []
    insideBarIndex = []

    for i, row in df.iterrows():

        if i == 0:
            activeDate.append(row.date)
            activeClose.append(row.close)
            activeOpen.append(row.open)
            activeHigh.append(row.high)
            activeLow.append(row.low)
            activeLowFirst.append(row.lowFirst)
            activeBarIndex.append(i)
            continue

        if (activeHigh[-1] > row.high) & (activeLow[-1] < row.low):
            insideDate.append(row.date)
            insideClose.append(row.close)
            insideOpen.append(row.open)
            insideHigh.append(row.high)
            insideLow.append(row.low)
            insideBarIndex.append(i)
            continue

        activeDate.append(row.date)
        activeClose.append(row.close)
        activeOpen.append(row.open)
        activeHigh.append(row.high)
        activeLow.append(row.low)
        activeLowFirst.append(row.lowFirst)
        activeBarIndex.append(i)

    # dfIgnoreInsideBars = pd.DataFrame(columns=['date', 'close', 'open', 'high', 'low'])

    noInsideBars = {
        'date': activeDate,
        'close': activeClose,
        'open': activeOpen,
        'high': activeHigh,
        'low': activeLow,
        'lowFirst': activeLowFirst,
        'barIndex': activeBarIndex,
    }
    insideBarsOnly = {
        'date': insideDate,
        'close': insideClose,
        'open': insideOpen,
        'high': insideHigh,
        'low': insideLow,
        'barIndex': insideBarIndex,
    }
    dfIgnoreInsideBars = pd.DataFrame.from_dict(noInsideBars)
    dfInsideBarsOnly = pd.DataFrame.from_dict(insideBarsOnly)

    # print(dfIgnoreInsideBars)
    # exit()
    # dfIgnoreInsideBars = pd.concat([df[:DELAY], dfIgnoreInsideBars])
    # dfIgnoreInsideBars = dfIgnoreInsideBars.reset_index(drop=True)

    # endregion INSIDE BARS

    # region: OUTSIDE BARS
    # dfIgnoreInsideBars['outside'] = (dfIgnoreInsideBars.high>dfIgnoreInsideBars.shift(1).high) & (dfIgnoreInsideBars.low<dfIgnoreInsideBars.shift(1).low)

    # print(dfIgnoreInsideBars)
    # exit()

    # endregion

    trendLine1, trendLine2, trendLine3 = getTrendLine(dfIgnoreInsideBars)

    #### TRENDLINE PROCESSING  END #####

    # topAndBottomPoints, HH_bars, LL_bars, HL_bars, LH_bars
    minorStuff = getTrendTopsAndBottoms(trendLine1, df)
    intermediateStuff = getTrendTopsAndBottoms(trendLine2, df)
    majorStuff = getTrendTopsAndBottoms(trendLine3, df)

    # (majorStuff['HH'] gives index of a top, and the number of bars to the next top.)

    # get little highs and lows
    lps = getLps(df)
    hps = getHps(df)

    lpsLows = lps[['date', 'low']].copy()
    hpsHighs = hps[['date', 'high']].copy()

    lpshps = pd.concat([lpsLows, hpsHighs], sort=True)
    lpshps = lpshps.sort_values(by=['date'])

    lps = Scatter(mode='markers', marker=dict(color='navy', size=10), x=lps.date, y=lps.low)
    hps = Scatter(mode='markers', marker=dict(color='navy', size=10), x=hps.date, y=hps.high)

    # plot H L data
    # minorHL_html = plotTopBotHist(minorStuff)
    # intermediateHL_html = plotTopBotHist(intermediateStuff)
    majorHL_html = plotTopBotHist(majorStuff)

    # region:trends
    minorUps, minorDowns, minorTopsBottoms, stringTrendMin, firstDateMin, lastDateMin, currTrendMin = trendFinder(
        minorStuff)
    intermediateUps, intermediateDowns, intermediateTopsBottoms, stringTrendInt, firstDateInt, lastDateInt, currTrendInt = trendFinder(
        intermediateStuff)

    majorUps, majorDowns, majorTopsBottoms, stringTrendMaj, firstDateMaj, lastDateMaj, currTrendMaj = trendFinder(
        majorStuff, barHeight=0.03)
    bigTopBotsMaj = majorTopsBottoms[(majorTopsBottoms.bigTop == True) | (majorTopsBottoms.bigBot == True)]
    # endregion

    # region:derivative trends

    firstDateList = [firstDateMaj]
    lastDateList = [lastDateMaj]
    bigTopBotsMajDrvd = bigTopBotsMaj

    firstDateMajDrvdFinal = None
    lastDateMajDrvdFinal = None

    for counter in range(11):
        majorUpsDrvd, majorDownsDrvd, majorTopsBottomsDrvd, stringTrendMajDrvd, firstDateMajDrvd, lastDateMajDrvd, currTrendMajDrvd = trendFinder(
            bigTopBotsMajDrvd, barHeight=0.03, upColor='navy', downColor='grey')

        if (firstDateMajDrvd != None) and (lastDateMajDrvd != None):
            firstDateList.append(firstDateMajDrvd)
            lastDateList.append(lastDateMajDrvd)
            bigTopBotsMajDrvd = majorTopsBottomsDrvd[
                (majorTopsBottomsDrvd.bigTop == True) | (majorTopsBottomsDrvd.bigBot == True)]
        else:
            date1 = df.iloc[df.high.idxmax].date
            date2 = df.iloc[df.low.idxmin].date

            if (date1 - date2).days > 0:
                firstDateMajDrvdFinal = date2
                lastDateMajDrvdFinal = date1
            else:
                firstDateMajDrvdFinal = date1
                lastDateMajDrvdFinal = date2
            break

    # endregion

    bigTopListInt = intermediateTopsBottoms[intermediateTopsBottoms.bigTop == True]
    bigBotListInt = intermediateTopsBottoms[intermediateTopsBottoms.bigBot == True]

    # bigTopListMaj = majorTopsBottoms[majorTopsBottoms.bigTop==True]
    # bigBotListMaj = majorTopsBottoms[majorTopsBottoms.bigBot == True]
    #
    # bigTops = Scatter(mode='markers',marker=dict(color='orange',size=10),x=bigTopListMaj.date,y=bigTopListMaj.point)
    # bigBots = Scatter(mode='markers', marker=dict(color='black', size=10), x=bigBotListMaj.date, y=bigBotListMaj.point)

    # region:Gann Angles

    # x0_date = lastDateInt
    # x0_idx = df[df.date == lastDateInt].index[0]
    # xLast_date = df.iloc[-1].date
    # xLast_idx = len(df)-1
    # y0 = float(trendLine2[trendLine2.date == lastDateInt].point)
    # scale = 0.004
    # trendUp = currTrendInt > 0
    #
    # intGann = [
    #     plotGannAngles(x0_date,x0_idx,xLast_date,xLast_idx,y0, trendUp =trendUp,ratio=0,scale=1,name='Flat',color='grey'),
    #     plotGannAngles(x0_date, x0_idx, xLast_date, xLast_idx, y0, trendUp =trendUp, ratio=1, scale=scale, name='1x1',color='orange'),
    #     plotGannAngles(x0_date, x0_idx, xLast_date, xLast_idx, y0, trendUp =trendUp, ratio=2, scale=scale, name='1x2',color='navy'),
    #     plotGannAngles(x0_date, x0_idx, xLast_date, xLast_idx, y0, trendUp =trendUp, ratio=4, scale=scale, name='1x4',color='gold'),
    #     plotGannAngles(x0_date, x0_idx, xLast_date, xLast_idx, y0, trendUp =trendUp, ratio=1.0/4, scale=scale, name='4x1',color='green'),
    #     plotGannAngles(x0_date, x0_idx, xLast_date, xLast_idx, y0, trendUp =trendUp, ratio=1.0/2, scale=scale, name='2x1',color='deepskyblue'),
    # ]
    # endregion

    # region: time retracement lines

    pointFirst = float(trendLine2[trendLine2.date == firstDateInt].point)
    pointLast = float(trendLine2[trendLine2.date == lastDateInt].point)
    intTimeRets = plotTimeRets(firstDateInt, lastDateInt,
                               maxHeight=max([pointFirst, pointLast]) * 1.05,
                               minHeight=min([pointFirst, pointLast]) * 0.95, )
    # endregion

    # region: charts for projection data
    # minorHL_html = trendProjector(minorTopsBottoms)
    # intermediateHL_html = trendProjector(intermediateTopsBottoms)
    # topProjHtml,botProjHtml,HHHtml,LHHtml,LLHtml,HLHtml = trendProjector(majorTopsBottoms, df.iloc[-1].date)

    # HprojInt, LprojInt, HprojFigInt, LprojFigInt, latestDateInt = trendProjector(intermediateTopsBottoms,
    #                                                                              df.iloc[-1].date, highColor='crimson',
    #                                                                              lowColor='pink')
    Hproj, Lproj, HprojFig, LprojFig, latestDateMaj = trendProjector(majorTopsBottoms, df.iloc[0].date)

    HprojFig['layout'].update(xaxis=dict(range=xaxisRange), showlegend=False)
    LprojFig['layout'].update(xaxis=dict(range=xaxisRange), showlegend=False)

    HprojFig = plotly.offline.plot(HprojFig,
                                             show_link=False,
                                             output_type='div',
                                             include_plotlyjs=False,
                                             # filename='minorHLData.html',
                                             auto_open=False,
                                             config={'displaylogo': False,
                                                     'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d',
                                                                                'zoomIn2d',
                                                                                'zoomOut2d',
                                                                                'resetScale2d', 'hoverCompareCartesian',
                                                                                'lasso2d'],
                                                     'displayModeBar': True
                                                     }),
    LprojFig = plotly.offline.plot(LprojFig,
                                             show_link=False,
                                             output_type='div',
                                             include_plotlyjs=False,
                                             # filename='minorHLData.html',
                                             auto_open=False,
                                             config={'displaylogo': False,
                                                     'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d',
                                                                                'zoomIn2d',
                                                                                'zoomOut2d',
                                                                                'resetScale2d', 'hoverCompareCartesian',
                                                                                'lasso2d'],
                                                     'displayModeBar': True
                                                     }),

    # endregion

    # region: Hurst Cycles
    # decision: hurst graph projection to always be in days resolution, so that it looks smooth when the period is small eg. 1.625 weeks
    hurst, avgList, hurstStartDate = mainHurst(df.drop(columns=['lowFirst']))
    avgList = []
    hurstProjs = []

    # if hurst > 2:
    #     for i in range(len(hurst)):
    #         length = len(hurst[i])
    #         thisBar = Bar(x=hurst[i], y=[5-i] * length,
    #                       # width=[barWidth]*length,
    #                       # xbins=dict(start=np.min(HH_bars), size=size, end=np.max(HH_bars)),
    #                       # hoverinfo='none',
    #                       # name='HL Projection from ' + eachProj.strftime("%y-%m-%d"),
    #                       dx=1, dy=1,
    #                       yaxis='y2',
    #                       # legendgroup='Proj of next lows',
    #                       showlegend=False,
    #                       opacity=0.4,
    #                       marker=dict(color='navy'),
    #                       hoverinfo='x',
    #                       )
    #         hurstProjs.append(thisBar)

    # print(hurstProjs)
    # exit()

    # find lowest point from 2008-09

    lowestBottom0809 = majorTopsBottoms[
        (majorTopsBottoms.bottom == True) & (majorTopsBottoms.date > datetime(2008, 1, 1))
        & (majorTopsBottoms.date < datetime(2009, 12, 31))]
    lowestBottom0809Date = lowestBottom0809[lowestBottom0809.point == min(lowestBottom0809.point)].date

    if len(avgList) == 0:
        hurstNominalList_weeks = [
            # years
            938.571,  # 18,
            469.286,  # 9,
            234.643,  # 4.5
            156.429,  # 3
            # months
            78.2144,  # 18
            52.1429,  # 12
            39.1072,  # 9
            # weeks
            26,  # 26
            13,  # 13
            6,  # 6.5
            3,  # 3.25
            1,  # 1.625
        ]

        nominalAmplitudes = [
            14, 13, 12, 11,
            9, 8, 7,
            5, 4, 3, 2, 1
        ]
        nonNomAmps = []

    else:
        hurstNominalList_weeks = [
            # years
            938.571,  # 18,
            469.286,  # 9,
            #    234.643 ,  # 4.5
            156.429,  # 3
            # months
            #    78.2144 ,  # 18
            52.1429,  # 12
            #    39.1072 ,  # 9
            # weeks
            # 26,  # 26
            # 13,  # 13
            6,  # 6.5
            3,  # 3.25
            1  # 1.625
        ]
        nominalAmplitudes = [
            14, 13, 11,
            8,
            3, 2, 1
        ]
        nonNomAmps = [
            12,
            9, 7,
            5, 4,
        ]

    # projecting nominal hurst dates
    #     hurstNomProj=[]
    # for cycleLength in hurstNominalList_weeks:
    #     thisCycleSet = []
    #     for i in range(1,lengthDf):
    #         projectedDate = lowestBottom0809Date + timedelta(weeks= i * cycleLength)
    #         if projectedDate < projLimitDate:
    #             thisCycleSet.append(projectedDate)
    #         else:
    #             break
    #     hurstNomProj.append(thisCycleSet)

    hurstTracesShort = []
    hurstTracesLong = []
    shortLongCutoff = 50 # weeks

    lowestBottom0809Date = pd.Timestamp(lowestBottom0809Date.values[0])

    if len(dfTail > 0):
        hurstX, hurstXInt = hurstSines(lowestPoint=lowestBottom0809Date, df=df, projLimitDate=endpoint)
    else:
        hurstX, hurstXInt = hurstSines(lowestPoint=lowestBottom0809Date, df=df, projLimitDate=endpoint)
    composite = np.zeros(len(hurstX))

    for idx in range(len(hurstNominalList_weeks)):
        sine = -1 * np.cos(2 * np.pi * hurstXInt / (hurstNominalList_weeks[idx] * 7)) * (nominalAmplitudes[idx])

        composite += sine

        hurstDf = pd.DataFrame(data=sine, columns=['point'])
        hurstDf['date'] = pd.to_datetime(hurstX)

        hurstSinePlot = Scatter(name='%i wks' % (hurstNominalList_weeks[idx]),
                                mode='lines', x=hurstDf.date, y=hurstDf.point,
                                hoverinfo='x+name')

        if hurstNominalList_weeks[idx] < shortLongCutoff:
            hurstTracesShort += [hurstSinePlot]
        else:
            hurstTracesLong += [hurstSinePlot]



    ####### do the non-nominal averages #######
    if len(avgList) > 0:
        hurstX, hurstXInt = hurstSines(lowestPoint=hurstStartDate, df=df, projLimitDate=endpoint)

        for idx in range(len(avgList)):
            sine = -1 * np.cos(2 * np.pi * hurstXInt / (avgList[idx] * 7)) * (nonNomAmps[idx])

            composite += sine

            hurstDf = pd.DataFrame(data=sine, columns=['point'])
            hurstDf['date'] = pd.to_datetime(hurstX)

            hurstSinePlot = Scatter(name='%i wks' % (avgList[idx]),
                                    mode='lines', x=hurstDf.date, y=hurstDf.point,
                                    hoverinfo='x+name')

            if avgList[idx] < shortLongCutoff:
                hurstTracesShort += [hurstSinePlot]
            else:
                hurstTracesLong += [hurstSinePlot]

    # composite plot
    hurstDf = pd.DataFrame(data=composite, columns=['point'])
    hurstDf['date'] = pd.to_datetime(hurstX)
    hurstCompositePlot = Scatter(name='Composite', mode='lines', x=hurstDf.date, y=hurstDf.point,
                                 line=dict(dash='dash'),
                                 # yaxis='y2',
                                 hoverinfo='x+name',
                                 )

    # hurstTraces += [hurstCompositePlot]

    # if len(dfTail)> 0: endpoint = dfTail.iloc[-1].date
    # else:
    # endpoint = df.iloc[-1].date + timedelta(weeks=12)


    hurstCompositeFig = Figure(data=[hurstCompositePlot], layout=Layout(xaxis=dict(range=xaxisRange), showlegend=False,title='Hurst Composite'))
    hurstShortFig = Figure(data=hurstTracesShort, layout=Layout(xaxis=dict(range=xaxisRange), showlegend=False,title='Hurst Short Cycles'))
    hurstLongFig = Figure(data=hurstTracesLong, layout=Layout(xaxis=dict(range=xaxisRange), showlegend=False,title='Hurst Long Cycles'))

    hurstCompositeHtml = plotly.offline.plot(hurstCompositeFig,
                                    show_link=False,
                                    output_type='div',
                                    include_plotlyjs=False,
                                    # filename='minorHLData.html',
                                    auto_open=False,
                                    config={'displaylogo': False,
                                            'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'zoomIn2d',
                                                                       'zoomOut2d',
                                                                       'resetScale2d', 'hoverCompareCartesian',
                                                                       'lasso2d'],
                                            'displayModeBar': True
                                            }),

    hurstShortHtml = plotly.offline.plot(hurstShortFig,
                                             show_link=False,
                                             output_type='div',
                                             include_plotlyjs=False,
                                             # filename='minorHLData.html',
                                             auto_open=False,
                                             config={'displaylogo': False,
                                                     'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d',
                                                                                'zoomIn2d',
                                                                                'zoomOut2d',
                                                                                'resetScale2d', 'hoverCompareCartesian',
                                                                                'lasso2d'],
                                                     'displayModeBar': True
                                                     }),

    hurstLongHtml = plotly.offline.plot(hurstLongFig,
                                             show_link=False,
                                             output_type='div',
                                             include_plotlyjs=False,
                                             # filename='minorHLData.html',
                                             auto_open=False,
                                             config={'displaylogo': False,
                                                     'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d',
                                                                                'zoomIn2d',
                                                                                'zoomOut2d',
                                                                                'resetScale2d', 'hoverCompareCartesian',
                                                                                'lasso2d'],
                                                     'displayModeBar': True
                                                     }),

    # endregion

    # region:retracement lines
    retLevels = []
    retLinesList = []
    colorList = ['red', 'navy', 'olive', 'orange', 'green', 'brown', 'purple', 'violet', 'khaki', 'silver', 'gold',
                 'blue', 'sienna']
    lowProjLevels = []
    # region:little retracements
    #
    #
    #
    # lastIndex = len(trendLine1) - 1
    # numRetracements = 10
    # minRetracements = []
    # for i in range(numRetracements):
    #     thisRet, thisRetLevels = retracementLines(trendLine1.iloc[lastIndex-i].point,trendLine1.iloc[lastIndex-i-1].point,
    #                                [trendLine1.iloc[lastIndex-i].date, endpoint],
    #                                name='mini-retrace '+str(i),
    #                                color=colorList[i])
    #     minRetracements += thisRet
    #     retLevels.append(thisRetLevels)
    # endregion

    # region:maj retracements
    #
    #
    majRetLines, majRetLevels, majLowProjLevels = retracementLines(majorTopsBottoms.iloc[-2].point,
                                                                   majorTopsBottoms.iloc[-1].point,
                                                                   [majorTopsBottoms.iloc[-2].date, endpoint],
                                                                   name='Maj Retrace',
                                                                   color=colorList[0],
                                                                   previousLowProjLevels=lowProjLevels
                                                                   )
    # retLevels.append(majRetLevels)
    # retLinesList.append(majRetLines)
    # lowProjLevels += majLowProjLevels

    for i in range(len(firstDateList)):
        majTrendRetLines, majTrendRetLevels, majTrendLowProjLevels = retracementLines(
            trendLine3[trendLine3.date == firstDateList[i]].point.values[0],
            trendLine3[trendLine3.date == lastDateList[i]].point.values[0],
            [firstDateList[i], endpoint],
            name='Maj Trend Retrace %i' % (i),
            color=colorList[(i + 1) % (len(colorList))],
            previousLowProjLevels=lowProjLevels
            )

        retLevels.append(majTrendRetLevels)
        retLinesList.append(majTrendRetLines)
        lowProjLevels += majTrendLowProjLevels

    highPoint1 = df[df.date == firstDateMajDrvdFinal].high.values[0]
    highPoint2 = df[df.date == lastDateMajDrvdFinal].high.values[0]
    lowPoint1 = df[df.date == firstDateMajDrvdFinal].low.values[0]
    lowPoint2 = df[df.date == lastDateMajDrvdFinal].low.values[0]
    firstPoint = None
    lastPoint = None

    if highPoint1 > highPoint2:
        firstPoint = highPoint1
        lastPoint = lowPoint2
    else:
        firstPoint = lowPoint1
        lastPoint = highPoint2

    majTrendRetLines, majTrendRetLevels, majTrendLowProjLevels = retracementLines(
        firstPoint,
        lastPoint,
        [firstDateMajDrvdFinal,
         endpoint],
        name='Maj Max Trend Retrace',
        color=colorList[(len(firstDateList) + 1) % (len(colorList))],
        previousLowProjLevels=lowProjLevels
    )

    retLevels.append(majTrendRetLevels)
    retLinesList.append(majTrendRetLines)
    lowProjLevels += majTrendLowProjLevels

    # remove duplicates from

    # endregion

    # retracement clusters
    # retClusters = getClusters(retLevels, df.iloc[-1].date, endpoint)

    clusters = clusterAlgo2(retLevels, df.iloc[-1].close, startX=df.iloc[-2].date, endX=endpoint)

    # endregion

    # region:signal tops
    intSignalTops = signalTops(df, bigTops=bigTopListInt)
    intSignalBots = signalBots(df, bigBots=bigBotListInt)
    # endregion

    #region Sun and Moon

    showStart = datetime(year=2004,month=7,day=7)

    sun = fixedIntervalBar(startDate=lowestBottom0809Date,endDate=endpoint,intervalDays=30,showStartDate=showStart)
    # sunFig = Figure(data=[sun],
    #                 layout=Layout(xaxis=dict(range=xaxisRange), showlegend=False,title='Sun',bargap=0.9),
    #                 )
    # sunHtml = plotly.offline.plot(sunFig,
    #                                          show_link=False,
    #                                          output_type='div',
    #                                          include_plotlyjs=False,
    #                                          # filename='minorHLData.html',
    #                                          auto_open=False,
    #                                          config={'displaylogo': False,
    #                                                  'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d',
    #                                                                             'zoomIn2d',
    #                                                                             'zoomOut2d',
    #                                                                             'resetScale2d', 'hoverCompareCartesian',
    #                                                                             'lasso2d'],
    #                                                  'displayModeBar': True
    #                                                  }),

    print('lowest point date:', lowestBottom0809Date)

    moon = fixedIntervalBar(startDate=lowestBottom0809Date, endDate=endpoint, intervalDays=7,
                           showStartDate=showStart)

    sunMoonXaxisRange = [
        showStart,
        endpoint
    ]
    moonFig = Figure(data=[sun, moon],
                     layout=Layout(xaxis=dict(range=sunMoonXaxisRange), showlegend=False,title='Sun + Moon Low Projections',
                                   bargap=0.5,
                                   barmode='stack',
                                   ),
                     )
    moonHtml = plotly.offline.plot(moonFig,
                                  show_link=False,
                                  output_type='div',
                                  include_plotlyjs=False,
                                  # filename='minorHLData.html',
                                  auto_open=False,
                                  config={'displaylogo': False,
                                          'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d',
                                                                     'zoomIn2d',
                                                                     'zoomOut2d',
                                                                     'resetScale2d', 'hoverCompareCartesian',
                                                                     'lasso2d'],
                                          'displayModeBar': True
                                          }),

    #endregion

    # region: active bars and inside bars and outside bars
    insideBars = Ohlc(name='Inside Bars', x=dfInsideBarsOnly.date, open=dfInsideBarsOnly.open,
                      close=dfInsideBarsOnly.close,
                      high=dfInsideBarsOnly.high, low=dfInsideBarsOnly.low,
                      opacity=0.5,
                      line=dict(width=1),
                      # hoverinfo='none',
                      # increasing=dict(line=dict(color= '#17BECF')),
                      # decreasing=dict(line=dict(color= '#17BECF')),
                      increasing=dict(line=dict(color='black')),
                      decreasing=dict(line=dict(color='black')),
                      )

    activeBars = Ohlc(name='Active Bars', x=dfIgnoreInsideBars.date, open=dfIgnoreInsideBars.open,
                      close=dfIgnoreInsideBars.close, high=dfIgnoreInsideBars.high, low=dfIgnoreInsideBars.low,
                      opacity=0.6,
                      line=dict(width=2.5),
                      # hoverinfo='none',
                      # hoverlabel=dict(namelengthsrc='none'),
                      # showlegend=False,
                      # increasing=dict(line=dict(color= '#17BECF')),
                      # decreasing=dict(line=dict(color= '#17BECF')),
                      increasing=dict(line=dict(color='black')),
                      decreasing=dict(line=dict(color='black')),
                      )

    dfOutsideHighFirst = dfIgnoreInsideBars[dfIgnoreInsideBars['outside'] & (~dfIgnoreInsideBars['lowFirst'])]
    dfOutsideLowFirst = dfIgnoreInsideBars[dfIgnoreInsideBars['outside'] & dfIgnoreInsideBars['lowFirst']]

    OHF = Ohlc(name='Outside High First', x=dfOutsideHighFirst.date, open=dfOutsideHighFirst.open,
               close=dfOutsideHighFirst.close, high=dfOutsideHighFirst.high, low=dfOutsideHighFirst.low,
               opacity=1,
               line=dict(width=2.5),
               increasing=dict(line=dict(color='black')),
               decreasing=dict(line=dict(color='black')),
               hoverinfo='none',
               )

    OLF = Ohlc(name='Outside Low First', x=dfOutsideLowFirst.date, open=dfOutsideLowFirst.open,
               close=dfOutsideLowFirst.close, high=dfOutsideLowFirst.high, low=dfOutsideLowFirst.low,
               opacity=1,
               line=dict(width=2.5),
               increasing=dict(line=dict(color='black')),
               decreasing=dict(line=dict(color='black')),
               hoverinfo='none',
               )
    # endregion

    # region:all bars
    allBars = Ohlc(name='All Bars', x=df.date, open=df.open,
                   close=df.close, high=df.high, low=df.low,
                   opacity=0.8,
                   line=dict(width=2.5),
                   # hoverinfo='none',
                   # hoverlabel=dict(namelengthsrc='none'),
                   # showlegend=False,
                   # increasing=dict(line=dict(color= '#17BECF')),
                   # decreasing=dict(line=dict(color= '#17BECF')),
                   # increasing=dict(line=dict(color='black')),
                   # decreasing=dict(line=dict(color='black')),
                   )
    # endregion

    # region:tail bars
    tailBars = Ohlc(name='Tail Bars', x=dfTail.date, open=dfTail.open,
                    close=dfTail.close, high=dfTail.high, low=dfTail.low,
                    opacity=0.5,
                    line=dict(width=2.5),
                    # hoverinfo='none',
                    # hoverlabel=dict(namelengthsrc='none'),
                    # showlegend=False,
                    # increasing=dict(line=dict(color= '#17BECF')),
                    # decreasing=dict(line=dict(color= '#17BECF')),
                    increasing=dict(line=dict(color='navy')),
                    decreasing=dict(line=dict(color='navy')),
                    )
    # endregion

    # region:trendlines
    # plot minor trendline points and lines
    # minor, minorTops, minorBottoms = plotTrendlines(trendLine1, minorStuff, name='Minor', color='grey', width=4)

    # plot intermediate trendline points and lines
    # intermediate, intermediateTops, intermediateBottoms = plotTrendlines(trendLine2, intermediateStuff,
    #                                                                      name='Intermediate', color='#000080', width=4)

    # plot major trendline points and lines
    major, majorTops, majorBottoms = plotTrendlines(trendLine3, majorStuff, name='Major', color='navy', width=3)
    # endregion

    # region:minorData to plot
    # minorData = [insideBars, activeBars,
    #              # OHF, OLF,
    #              minor, minorTops, minorBottoms,
    #              minorUps, minorDowns
    #              ]
    # endregion

    # region: intermediate data to plot
    # intermediateData = [
    #     # insideBars,
    #     activeBars,
    #     # OHF, OLF,
    #     intermediate, intermediateTops, intermediateBottoms,
    #     intermediateUps, intermediateDowns
    # ]
    # intGann +
    # intermediateData += intRetLines
    # intermediateData += intTimeRets
    # intSignalTops + intSignalBots
    # + [bigTops,bigBots]
    # endregion

    # region: major data to plot
    majorData = [
        # insideBars,
        # minor,
        # minorTops, minorBottoms,
        # activeBars,
        # OHF, OLF,
        # intermediate,
        allBars, tailBars,
        # major,
        # majorTops, majorBottoms,

        # majorUps, majorDowns,
        # majorUpsDrvd, majorDownsDrvd,
        # lps,hps,
    ]

    # for eachRet in retLinesList:
    #     majorData += eachRet
    #
    majorData += clusters

    tops = Scatter(x=df.date, y=df.high, mode='lines')
    bots = Scatter(x=df.date, y=df.low, mode='lines')

    # majorData += [hurstCompositePlot]
    # majorData+= [tops,bots]
    #
    # majorData += retClusters
    # majorData += Hproj
    # majorData += Lproj
    # majorData += minRetracements

    # endregion

    # region:layouts
    # layoutMin = Layout(
    #     title='EUR/USD Weekly<br>' + stringTrendMin,
    #     xaxis=dict(
    #         rangeslider=dict(
    #             visible=False
    #         ),
    #         showgrid=True,
    #     ),
    #     showlegend=False,
    #     # annotations=minorAnnot
    # )
    #
    # layoutInt = Layout(
    #     title='EUR/USD Weekly<br>' + stringTrendInt,
    #     xaxis=dict(
    #         rangeslider=dict(
    #             visible=False
    #         ),
    #         showgrid=True,
    #     ),
    #     showlegend=False,
    #     yaxis=dict(range=[-0.1, max(df.high) * 1.2])
    #     # annotations=minorAnnot
    # )

    layoutMaj = Layout(
        title='EUR/USD Weekly<br>' + stringTrendMaj,
        xaxis=dict(
            rangeslider=dict(
                visible=False

            ),
            range=xaxisRange,
            showgrid=True,

        ),
        showlegend=True,
        yaxis=dict(range=[min(df.low) * 0.8, max(df.high) * 1.2]),
        # paper_bgcolor=backgroundColor,
        # plot_bgcolor=backgroundColor,
        # barmode='stack',
        # yaxis2=dict(overlaying='y', side='right', range=[min(composite) * 0.8, max(composite) * 1.2]),
        # annotations=majRetAnnot
    )
    # endregion

    # minorFig = Figure(data=minorData, layout=layoutMin)
    # intermediateFig = Figure(data=intermediateData, layout=layoutInt)
    majorFig = Figure(data=majorData, layout=layoutMaj)

    majorTrendFig = Figure(data=[major, majorTops, majorBottoms], layout = layoutMaj)
    majorTrendHtml = plotly.offline.plot(majorTrendFig,
                        show_link=False,
                        output_type='div',
                        include_plotlyjs=False,
                        # filename='minorHLData.html',
                        auto_open=False,
                        config={'displaylogo': False,
                                'modeBarButtonsToRemove': ['sendDataToCloud', 'select2d', 'zoomIn2d',
                                                           'zoomOut2d',
                                                           'resetScale2d', 'hoverCompareCartesian',
                                                           'lasso2d'],
                                'displayModeBar': True
                                }),

    # plotly.offline.plot(fig,image='png',image_filename='3trends',image_width=7200,image_height=1200)

    # plotter(minorFig,'minorTrend_daily.html', [minorHL_html])
    # plotter(intermediateFig, 'intermediateTrend_daily.html', [intermediateHL_html])


    # htmls = [majorTrendHtml] + hurstHtml

    plotter(majorFig, 'majorTrend_weeklyFrom2008.html',
            HprojFig+
            LprojFig+
            majorTrendHtml+
            hurstCompositeHtml+
            hurstShortHtml+
            hurstLongHtml+
            # sunHtml+
            moonHtml)
    #         # [topProjHtml, botProjHtml, HHHtml, LHHtml, LLHtml, HLHtml],
    #         )

    # help(plotly.offline.plot)

    # print(HprojInt)
    # exit()
    #
    # verticalPlot(mainTrace=majorData,others=hurstTraces)
    #              # hurst2=hurstProjs[1],hurst3=hurstProjs[2],
    #              # hurst4=hurstProjs[3],hurst5=hurstProjs[4],)

    endTime = time.time()
    elapsed = endTime - startTime
    print("Operation took a total of %.2f seconds." % (elapsed))


    return majorFig

if __name__ == '__main__':

    main()