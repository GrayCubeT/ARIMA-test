import pandas
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


dataFrame = pandas.read_excel(io='Данные.xlsx')

dataFrame.reset_index(inplace=True)
dataFrame['Date'] = pandas.to_datetime(dataFrame['Date'])
dataFrame = dataFrame.set_index('Date')

model = ARIMA(dataFrame.Value, order=(1, 1, 4)).fit()
print(model.aic)
print(model.predict(start='1989-01-01', end='1993-12-01', dynamic=False))