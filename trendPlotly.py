from trendFunctions import *

startTime = time.time()

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