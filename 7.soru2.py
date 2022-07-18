# Data manipulation

import numpy as np
import pandas as pd
# Plots

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plt.style.use('fivethirtyeight')

# Modelling and Forecasting

from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from sklearn.metrics import mean_absolute_percentage_error
# Warnings configuration
#Data Hazırlanması
import warnings
warnings.filterwarnings('ignore')
data1=pd.read_csv("Hourly_Load.csv")
data=data1.copy()
data.head()
data

dt=data.copy()
dt['Datetime'] = pd.to_datetime(dt['Datetime'], format='%Y-%m-%dT%H:%M:%S')
dt = dt.set_index('Datetime')
dt = dt.asfreq('60min')
dt = dt.sort_index()
dt
(dt.index == pd.date_range(start=dt.index.min(),
                              end=dt.index.max(),
                              freq=dt.index.freq)).all()
dt

dt.info()
print(dt.isnull().sum())
#Filling the Missing Values – Imputation

updated_dt = dt
updated_dt['PJM_Load_MW']=updated_dt['PJM_Load_MW'].fillna(updated_dt['PJM_Load_MW'].mean())
updated_dt.info()

print(updated_dt.isnull().sum())
print(dt.isnull().sum())
dt
updated_dt



# Split data into train-val-test
# ==============================================================================
dt = dt.loc['1998-12-31 01:00:00': '2001-01-02 00:00:00']
end_train = '1999-12-31 23:00:00'
end_validation = '2000-11-30 23:59:00'
dt_train = dt.loc[: end_train, :]
dt_val   = dt.loc[end_train:end_validation, :]
dt_test  = dt.loc[end_validation:, :]

print(f"Train dates      : {dt_train.index.min()} --- {dt_train.index.max()}")
print(f"Validation dates : {dt_val.index.min()} --- {dt_val.index.max()}")
print(f"Test dates       : {dt_test.index.min()} --- {dt_test.index.max()}")




# Time series plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 4))
dt_train.PJM_Load_MW.plot(ax=ax, label='train', linewidth=1)
dt_val.PJM_Load_MW.plot(ax=ax, label='validation', linewidth=1)
dt_test.PJM_Load_MW.plot(ax=ax, label='test', linewidth=1)
ax.set_title('elektrik tüketimi')
ax.legend();




# Boxplot for annual seasonality
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 3.5))
dt['month'] = dt.index.month
dt.boxplot(column='PJM_Load_MW', by='month', ax=ax,)
dt.groupby('month')['PJM_Load_MW'].median().plot(style='o-', linewidth=0.8, ax=ax)
ax.set_ylabel('PJM_Load_MW')
ax.set_title('Consume distribution by month')
fig.suptitle('');


# Boxplot for weekly seasonality
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 3.5))
dt['week_day'] = dt.index.day_of_week + 1
dt.boxplot(column='PJM_Load_MW', by='week_day', ax=ax)
dt.groupby('week_day')['PJM_Load_MW'].median().plot(style='o-', linewidth=0.8, ax=ax)
ax.set_ylabel('PJM_Load_MW')
ax.set_title('Consume distribution by week day')
fig.suptitle('');






# Boxplot for daily seasonality
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 3.5))
dt['hour_day'] = dt.index.hour + 1
dt.boxplot(column='PJM_Load_MW', by='hour_day', ax=ax)
dt.groupby('hour_day')['PJM_Load_MW'].median().plot(style='o-', linewidth=0.8, ax=ax)
ax.set_ylabel('PJM_Load_MW')
ax.set_title('Consume distribution by the time of the day')
fig.suptitle('');









    # Create and train forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor = make_pipeline(StandardScaler(), Ridge()),
                lags      = 24
             )

forecaster.fit(y=dt.loc[:end_validation, 'PJM_Load_MW'])
forecaster
 



# Backtest with test data and prediction intervals
# ==============================================================================

metric, predictions = backtesting_forecaster(
                            forecaster = forecaster,
                            y          = dt.PJM_Load_MW,
                            initial_train_size = len(dt.PJM_Load_MW[:end_validation]),
                            
                            steps      = 24,
                            metric     = 'mean_absolute_percentage_error'*100,
                            interval            = [10, 90],
                            n_boot              = 500,
                            in_sample_residuals = True,
                            verbose             = False
                       )
print('Backtesting metric:', metric*100)
# Plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 3.5))
dt.loc[predictions.index, 'PJM_Load_MW'].plot(linewidth=2, label='real', ax=ax)
predictions.iloc[:, 0].plot(linewidth=2, label='prediction', ax=ax)
ax.set_title('Prediction vs real Consume')
ax.fill_between(
    predictions.index,
    predictions.iloc[:, 1],
    predictions.iloc[:, 2],
    alpha = 0.3,
    color = 'red',
    label = 'prediction interval' 
)
ax.legend();

beklenen 



