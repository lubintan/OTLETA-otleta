
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt

from math import pi
from datetime import datetime

import pandas as pd
import numpy as np


from bokeh.plotting import figure, show, output_file
from bokeh.models import DatetimeTickFormatter, HoverTool
from bokeh.models.sources import ColumnDataSource

# TITLE = 'TrendLine Delay = ' + str(DELAY)
TITLE = 'minor-grey, intermediate-blue, major-black'

def getTrendLine(dfIgnoreInsideBars, DELAY):
    trendPoints = []

    # dfIgnoreInsideBars = pd.DataFrame(df)

    for index, row in dfIgnoreInsideBars.iterrows():
        if index < DELAY:
            continue

        # TODO: outside bars (if row.high > prevDfHigh and row.low < prevDfLow)

        if row.outside:
            if len(trendPoints) >= 2:
                if trendPoints[-1][1] >= trendPoints[-2][1]:  # trending up
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

                elif trendPoints[-1][1] < trendPoints[-2][1]:  # trending down
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
            continue

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
            trendPoints.append([row.date, row.low])
        elif trendHigh:
            trendPoints.append([row.date, row.high])

        # otherwise do nothing

    trendLine = pd.DataFrame(trendPoints, columns=['date', 'point'])

    return trendLine


if __name__ == '__main__':

    df = pd.read_excel('EURUSD Weekly Data for Swing Indicator.xlsx')
    # df = pd.read_csv('EURUSD Weekly Data for Swing Indicator.csv')

    # df = pd.read_excel('test1.xlsx')



    # Convert Date Format
    df.columns = ['date','close','open','high','low']
    df['date'] = pd.to_datetime(df['date'])





    ######## Group by Months ###############
    #region Group By Months
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

    #check open and close
    # checkMonth = 7
    # checkYear= 2017
    # print(df[(df.date.dt.month==checkMonth) & (df.date.dt.year==checkYear)])
    # print(months[(months.month==checkMonth) & (months.year==checkYear)])
    # exit()


    df = pd.DataFrame(months)
    #endregion
    ##### End group by months#####

################### End convert to months section #################

#TODO:
#     - write program to be efficient: only make changes to new data collected (don’t recompute everything.)
# - designer
# - see swing high and low numbers. High green circle, low red circle.

#inside bars (fully ignore for trend line calculation)
#region: INSIDE BARS
    # chunk of older inside bars code that only compares to previous bar
    # #create dataframe without inside bars for trend line calculation
    # dfWoInitial = pd.DataFrame(df[DELAY:])
    #
    #
    # dfIgnoreInsideBars = dfWoInitial[~((dfWoInitial.high<dfWoInitial.shift(1).high) & (dfWoInitial.low>dfWoInitial.shift(1).low))]
    # dfIgnoreInsideBars = pd.concat([df[:DELAY], dfIgnoreInsideBars])
    # dfIgnoreInsideBars = dfIgnoreInsideBars.reset_index(drop=True)

###########################################################

    # dfWoInitial = pd.DataFrame(df[DELAY:])



    activeDate=[]
    activeClose=[]
    activeOpen=[]
    activeHigh=[]
    activeLow=[]
    activeLowFirst=[]

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

    dfIgnoreInsideBars = pd.DataFrame.from_dict(noInsideBars)

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

########### Minor Trendline Processing ########
    trendLine1 = getTrendLine(dfIgnoreInsideBars,1)
    trendLine2 = getTrendLine(dfIgnoreInsideBars, 2)
    trendLine3 = getTrendLine(dfIgnoreInsideBars, 3)
#### TRENDLINE PROCESSING  END #####


    TOOLS = "pan, tap,box_zoom,reset,save"
    #hover, crosshair...



    p = figure(tools=TOOLS, title=TITLE)
    p.sizing_mode = 'stretch_both'
    p.xaxis.major_label_orientation = pi / 4
    # df['dateStrings'] = df['date'].dt.strftime('%d-%b-%Y')
    # # print(dateStrings.values.tolist())
    # p.xaxis.ticker = dateStrings.values.tolist()
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.grid.grid_line_alpha = 0.7
    p.toolbar.logo = None
    p.xaxis.formatter = DatetimeTickFormatter(
        days=["%d-%b-%Y"]

    )

    # map dataframe indices to date strings and use as label overrides
    p.xaxis.major_label_overrides = {
        i: date.strftime('%d-%b-%Y') for i, date in enumerate(df["date"])
    }

    # tickers = pd.to_datetime(df.index.values).astype(int) / 10 ** 6
    # p.xaxis.ticker = FixedTicker(ticks = tickers)


    # plot all bars
    p.segment(df.date, df.high, df.date, df.low, color="#40E0D0", line_width=2)

    # overlay non-inside bars
    p.segment(dfIgnoreInsideBars.date, dfIgnoreInsideBars.high, dfIgnoreInsideBars.date, dfIgnoreInsideBars.low, color="#000080", line_width=3)

    # overlay outside bars
    dfOutsideHighFirst = dfIgnoreInsideBars[dfIgnoreInsideBars['outside'] & (~dfIgnoreInsideBars['lowFirst'])]
    dfOutsideLowFirst = dfIgnoreInsideBars[dfIgnoreInsideBars['outside'] & dfIgnoreInsideBars['lowFirst']]
    p.segment(dfOutsideHighFirst.date, dfOutsideHighFirst.high, dfOutsideHighFirst.date, dfOutsideHighFirst.low, color="#000080", line_width=5)
    p.segment(dfOutsideLowFirst.date, dfOutsideLowFirst.high, dfOutsideLowFirst.date, dfOutsideLowFirst.low, color="#0000FF", line_width=5)



    # #open close candlesticks
    # w = 15*24 * 60 * 60 * 1000  # full day in ms
    # # w = 0.7
    # inc = df.close >= df.open
    # dec = df.close < df.open
    # p.vbar(x=df.date[inc], width=w, top=df.open[inc], bottom=df.close[inc], fill_color="green", line_color="green",
    #        line_alpha=1)
    # p.vbar(x=df.date[dec], width=w, top=df.open[dec], bottom=df.close[dec], fill_color="red", line_color="red",
    #        line_alpha=1)

    # plot minor trendline points and lines
    p.line(trendLine1.date, trendLine1.point, color='grey', line_width=2)
    p.circle(trendLine1.date, trendLine1.point, color='grey', size=10)

    # plot intermediate trendline points and lines
    p.line(trendLine2.date, trendLine2.point, color='blue', line_width=3)
    p.circle(trendLine2.date, trendLine2.point, color='blue', size=10)

    # plot major trendline points and lines
    p.line(trendLine3.date, trendLine3.point, color='black', line_width=4)
    p.circle(trendLine3.date, trendLine3.point, color='black', size=10)

    output_file("candlestick.html", title="candlestick.py example")

    show(p)  # open a browser
