import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# find the percentages of each age group
TOTAL = 5118136.0
UNDER20s = 105770/TOTAL
UNDER30s = 1170176/TOTAL
UNDER40s = 1797410/TOTAL
UNDER50s = 1256255/TOTAL
UNDER60s = 692903/TOTAL
UNDER65s = 88260/TOTAL
ABOVE65s = 7362/TOTAL



def generateSamplesAndFindMedian(sizeUnder20s, sizeUnder30s, sizeUnder40s, sizeUnder50s, sizeUnder60s, sizeUnder65s, sizeAbove65s):

    sampleUnder20s = np.random.normal(17487, 7459, sizeUnder20s)
    sampleUnder30s = np.random.normal(32481, 18207, sizeUnder30s)
    sampleUnder40s = np.random.normal(47044, 37225, sizeUnder40s)
    sampleUnder50s = np.random.normal(56863, 65919, sizeUnder50s)
    sampleUnder60s = np.random.normal(59514, 103232, sizeUnder60s)
    sampleUnder65s = np.random.normal(76572, 137916, sizeUnder65s)
    sampleAbove65s = np.random.normal(114779, 360839, sizeAbove65s)

    totalSample = np.concatenate([sampleUnder20s,sampleUnder30s,sampleUnder40s,sampleUnder50s,sampleUnder60s,sampleUnder65s,sampleAbove65s])

    median = np.median(totalSample)

    return median

if __name__ == '__main__':


    ############# Question 2a ################

    totalSampleSize = 200

    sizeUnder20s = int(UNDER20s * totalSampleSize)
    sizeUnder30s = int(UNDER30s * totalSampleSize)
    sizeUnder40s = int(UNDER40s * totalSampleSize)
    sizeUnder50s = int(UNDER50s * totalSampleSize)
    sizeUnder60s = int(UNDER60s * totalSampleSize)
    sizeUnder65s = int(UNDER65s * totalSampleSize)
    sizeAbove65s = totalSampleSize - sizeUnder20s - sizeUnder30s - sizeUnder40s - sizeUnder50s - sizeUnder60s - sizeUnder65s

    median = generateSamplesAndFindMedian(sizeUnder20s, sizeUnder30s, sizeUnder40s, sizeUnder50s, sizeUnder60s, sizeUnder65s, sizeAbove65s)

    print('Answer to Question 2a')
    print('Median =',median)

    ############# Question 2b ################

    numberOfRepeats = 5000
    listOfMedians = []

    for eachRepeat in range(numberOfRepeats):
        median = generateSamplesAndFindMedian(sizeUnder20s, sizeUnder30s, sizeUnder40s, sizeUnder50s, sizeUnder60s,
                                              sizeUnder65s, sizeAbove65s)
        listOfMedians.append(median)


    medianChart = pd.Series(listOfMedians)

    print()
    print('Question 2b - see chart.')
    print('Mean of Medians =',np.array(listOfMedians).mean())
    print('Std Dev of Medians =', np.array(listOfMedians).std())

    ax = medianChart.hist()
    ax.set(xlabel='', ylabel='Median Value', title='Distribution of Medians, Repeats = 5000, Sample Size = 200')
    plt.show()


    ############# Question 2c - Size = 500 ################
    totalSampleSize = 500

    sizeUnder20s = int(UNDER20s * totalSampleSize)
    sizeUnder30s = int(UNDER30s * totalSampleSize)
    sizeUnder40s = int(UNDER40s * totalSampleSize)
    sizeUnder50s = int(UNDER50s * totalSampleSize)
    sizeUnder60s = int(UNDER60s * totalSampleSize)
    sizeUnder65s = int(UNDER65s * totalSampleSize)
    sizeAbove65s = totalSampleSize - sizeUnder20s - sizeUnder30s - sizeUnder40s - sizeUnder50s - sizeUnder60s - sizeUnder65s

    numberOfRepeats = 5000
    listOfMedians = []

    for eachRepeat in range(numberOfRepeats):
        median = generateSamplesAndFindMedian(sizeUnder20s, sizeUnder30s, sizeUnder40s, sizeUnder50s, sizeUnder60s,
                                              sizeUnder65s, sizeAbove65s)
        listOfMedians.append(median)


    medianChart = pd.Series(listOfMedians)

    print()
    print('Question 2c, Sample Size = 500 - see chart.')
    print('Mean of Medians =',np.array(listOfMedians).mean())
    print('Std Dev of Medians =', np.array(listOfMedians).std())

    ax = medianChart.hist()
    ax.set(xlabel='', ylabel='Median Value', title='Distribution of Medians, Repeats = 5000, Sample Size = 500')
    plt.show()

    ############# Question 2c - Size = 1000 ################
    totalSampleSize = 1000

    sizeUnder20s = int(UNDER20s * totalSampleSize)
    sizeUnder30s = int(UNDER30s * totalSampleSize)
    sizeUnder40s = int(UNDER40s * totalSampleSize)
    sizeUnder50s = int(UNDER50s * totalSampleSize)
    sizeUnder60s = int(UNDER60s * totalSampleSize)
    sizeUnder65s = int(UNDER65s * totalSampleSize)
    sizeAbove65s = totalSampleSize - sizeUnder20s - sizeUnder30s - sizeUnder40s - sizeUnder50s - sizeUnder60s - sizeUnder65s

    numberOfRepeats = 5000
    listOfMedians = []

    for eachRepeat in range(numberOfRepeats):
        median = generateSamplesAndFindMedian(sizeUnder20s, sizeUnder30s, sizeUnder40s, sizeUnder50s, sizeUnder60s,
                                              sizeUnder65s, sizeAbove65s)
        listOfMedians.append(median)

    medianChart = pd.Series(listOfMedians)

    print()
    print('Question 2c, Sample Size = 1000 - see chart.')
    print('Mean of Medians =', np.array(listOfMedians).mean())
    print('Std Dev of Medians =', np.array(listOfMedians).std())

    ax = medianChart.hist()
    ax.set(xlabel='', ylabel='Median Value', title='Distribution of Medians, Repeats = 5000, Sample Size = 1000')
    plt.show()