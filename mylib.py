import pandas
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import r2_score

def seasonal_decompose_plots(dataFrame, model):
    '''
    Plot Trend, seasonal and residue decomposition
    '''
    res = sm.tsa.seasonal_decompose(dataFrame["Value"], model=model)

    plt.figure(figsize=(24,10))
    plt.subplot(311)
    plt.title("Trend", rotation='vertical', x=-0.05,y=0.5)
    plt.plot(res.trend)
    plt.subplot(312)
    plt.title("Seasonal", rotation='vertical', x=-0.05,y=0.5)
    plt.plot(res.seasonal)
    plt.subplot(313)
    plt.title("Residue", rotation='vertical', x=-0.05,y=0.5)
    plt.plot(res.resid)
    plt.show()

def autocorrelation_plots(dataFrame, shift=1):
    '''
    Plot both autocorrelation and partial autocorrelation functions
    '''
    diffTimeSeries = dataFrame.diff(periods=shift).dropna()
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(diffTimeSeries.Value.values.squeeze(), 
                                   lags=25, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(diffTimeSeries.Value, 
                                    lags=25, ax=ax2, method='ywm')
    plt.show()

def default_reader(filename):
    '''
    Reading and adding time indexes
    '''
    df = pandas.read_excel(io=filename, engine='openpyxl')
    df.reset_index(inplace=True)
    df['Date'] = pandas.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df

def predict(model, answersData):
    '''
    Predict using a model provided
    answersData is updated with a "Predict" column
    plots the data alongside answersData 
    '''
    answersData["Predict"] = model.predict(start='1989-01-01', end='1993-12-01', dynamic=False)
    plt.figure(figsize=(16,8))
    plt.plot(answersData.Value, label='Answers')
    plt.plot(answersData["Predict"], label='Predicted')
    plt.legend()
    plt.show()
    print("r2 score: ", 
          r2_score(answersData['Value'], answersData['Predict']))
    return

def check_periodic(dataFrame, k, printAns=False, plotting=False): 
    ''' A function for checking a hypothesis that
    a time series that is a difference of period k
    of the provided time series is stationary
    
    Arguments:
    dataframe -- time series to use
    k -- shift
    
    Keyword arguments:
    printAns -- if True, prints the answer
    plotting -- if True, plots the found stationary time series
    
    Returns: 
    probability of the resulting time series being non-stationary
    ''' 
    dataFrame = pandas.Series(dataFrame)
    check = None
    if k <= 0:
        check = dataFrame
    else:
        check = dataFrame.diff(periods=k).dropna()
    test = sm.tsa.adfuller(check)
    # using 0.01 significance
    if test[1] > 0.01:
        if printAns:
            print('p-value of Dickey-Fuller test: p={0:.4f}'.format(test[1]))
            print('Dickey-Fuller test result: False') 
            return
        # no plotting is done on non-stationary result
        return test[1]
    else:
        if plotting:
            check.plot(figsize=(12,6))
            plt.show()
        if printAns:
            print('p-value of Dickey-Fuller test: p={0:.4f}'.format(test[1]))
            print('Dickey-Fuller test result: True')
            return
        return test[1]


def moving_statistics(dataFrame):
    '''
    Plot all the moving statistics plots and show them
    '''
    rollingSeries = dataFrame['Value'].rolling(window=100)
    expandingSeries = dataFrame['Value'].expanding(20)
    weightedSeries = dataFrame['Value'].ewm(com=20)

    plt.figure(figsize=(24,6))
    plt.subplot(131)
    plt.title('simple moving statistics')
    plt.plot(rollingSeries.mean() + rollingSeries.std(), 
             color='orange', label = 'Moving average')
    plt.plot(rollingSeries.mean() - rollingSeries.std(), 
             color='orange')
    plt.plot(dataFrame['Value'], label='Values')
    plt.legend(fontsize=8)

    plt.subplot(132)
    plt.title('expanding moving statistics')
    plt.plot(expandingSeries.mean() + expandingSeries.std(), 
             color='orange', label = 'Moving average')
    plt.plot(expandingSeries.mean() - expandingSeries.std(), 
             color='orange')
    plt.plot(dataFrame['Value'], label='Values')
    plt.legend(fontsize=8)

    plt.subplot(133)
    plt.title('wighted moving statistics')
    plt.plot(weightedSeries.mean() + weightedSeries.std(), 
             color='orange', label = 'Moving average')
    plt.plot(weightedSeries.mean() - weightedSeries.std(), 
             color='orange')
    plt.plot(dataFrame['Value'], label='Values')
    plt.legend(fontsize=8)