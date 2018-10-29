import pandas as pd
import matplotlib.pyplot as plt
import datetime
from mpl_toolkits.mplot3d import Axes3D

def getNumberOfPatientsByHour(hospital_data):
    #create a list with 24 elements (1 for each hour)
    hourMark = range(23) # this gives [0, 1, 2, ... , 22]
    patientCount = []


    for hour in hourMark:
        counter = 0

        for patientTime in hospital_data['REGIS_TIME']:
            if (patientTime >= datetime.time(hour=hour)) and (patientTime < datetime.time(hour=(hour+1))): # this will give us between the hours of 00:00 to 22:59
                counter = counter + 1

        patientCount.append(counter)

    #now account for the patients who came in between 23:00 to 23:59
    patientCount.append(len(hospital_data["REGIS_TIME"].tolist()) - sum(patientCount))

    return patientCount

if __name__ == '__main__':

    ###### Question 1a #####
    # I believe this is asking for a chart of the number of patients by hour, regardless of the day.
    # So we need to plot Number of Patients (y-axis) vs Hour of the day (x-axis).

    #read the csv file
    hospital_data = pd.read_csv('Hospital.csv')

    #convert the REGIS_TIME format from string to timef
    hospital_data['REGIS_TIME'] = pd.to_datetime(hospital_data['REGIS_TIME'], format='%H:%M:%S').dt.time

    patientCount = getNumberOfPatientsByHour(hospital_data)

    # print(patientCount)

    hourLabels  = ['00:00-00:59']
    hourLabels += ['01:00-01:59']
    hourLabels += ['02:00-02:59']
    hourLabels += ['03:00-03:59']
    hourLabels += ['04:00-04:59']
    hourLabels += ['05:00-05:59']
    hourLabels += ['06:00-06:59']
    hourLabels += ['07:00-07:59']
    hourLabels += ['08:00-08:59']
    hourLabels += ['09:00-09:59']
    hourLabels += ['10:00-10:59']
    hourLabels += ['11:00-11:59']
    hourLabels += ['12:00-12:59']
    hourLabels += ['13:00-13:59']
    hourLabels += ['14:00-14:59']
    hourLabels += ['15:00-15:59']
    hourLabels += ['16:00-16:59']
    hourLabels += ['17:00-17:59']
    hourLabels += ['18:00-18:59']
    hourLabels += ['19:00-19:59']
    hourLabels += ['20:00-20:59']
    hourLabels += ['21:00-21:59']
    hourLabels += ['22:00-22:59']
    hourLabels += ['23:00-23:59']

    # print(hourLabels)

    patientChart = pd.DataFrame({'val':patientCount}, index=hourLabels)
    # patientChart = pd.Series(patientCount)

    ax = patientChart.plot.bar(color='blue',rot=45, legend=False,figsize=(20,25), grid=True)
    ax.set(xlabel='', ylabel='Number of Registrations', title='')
    plt.show()



    ###### Question 1b ######

    print('Printing graph for Question 1b...')

    # Added a column to the data to simply say which day of the week it is. (Monday = Day 0, Sunday = Day 6)
    hospital_data['DAY'] = pd.to_datetime(hospital_data['REGIS_DATE']).dt.dayofweek

    # Filter out the data by day o the week
    mondayData = hospital_data.query('DAY == 0')
    tuesdayData = hospital_data.query('DAY == 1')
    wednesdayData = hospital_data.query('DAY == 2')
    thursdayData = hospital_data.query('DAY == 3')
    fridayData = hospital_data.query('DAY == 4')
    saturdayData = hospital_data.query('DAY == 5')
    sundayData = hospital_data.query('DAY == 6')

    # For each day of the week, do the same thing to the data as Question 1a.
    # i.e. split the patient registration count by hour of the day.
    mondayPatientCount = getNumberOfPatientsByHour(mondayData)
    tuesdayPatientCount = getNumberOfPatientsByHour(tuesdayData)
    wednesdayPatientCount = getNumberOfPatientsByHour(wednesdayData)
    thursdayPatientCount = getNumberOfPatientsByHour(thursdayData)
    fridayPatientCount = getNumberOfPatientsByHour(fridayData)
    saturdayPatientCount = getNumberOfPatientsByHour(saturdayData)
    sundayPatientCount = getNumberOfPatientsByHour(sundayData)

    # Put all the data into 1 data frame.
    # Not sure if this is exactly how the question wants the results to be presented. Can check with your friends, or prof?
    patientCountByDay = pd.DataFrame({'0 - Mon':mondayPatientCount,'1 - Tue':tuesdayPatientCount,'2 - Wed':wednesdayPatientCount,'3 - Thu':thursdayPatientCount,'4 - Fri':fridayPatientCount,'5 - Sat':saturdayPatientCount,'6 - Sun':sundayPatientCount}, index = hourLabels)
    ax = patientCountByDay.plot.bar(subplots=True, figsize=(20,25), legend=False, sharey=True, rot=45, grid=True)

    plt.show()