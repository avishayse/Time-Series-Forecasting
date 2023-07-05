import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Define the symbol of the stock
symbol = 'AAPL'

# Download the data
df = yf.download(symbol)

# Use only 'Open' column
df = df[['Open']]

# Drop missing values
df = df.dropna()

# Split the data into train and test
size = int(len(df) * 0.8)
df_train, df_test = df.iloc[0:size], df.iloc[size:len(df)]

# To avoid the error related to date index, we'll reset the index.
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Fit the model
# (p,d,q) are parameters for ARIMA. These would ideally be selected through exploration and optimization.
model = ARIMA(df_train, order=(5,1,0))
model_fit = model.fit()

# Make predictions
predictions = model_fit.predict(start=len(df_train), end=len(df_train) + len(df_test)-1, dynamic=False)

# Print lengths of test data and predictions
print("Length of test data:", len(df_test))
print("Length of predictions:", len(predictions))

# Cut off the extra predictions if necessary
if len(predictions) > len(df_test):
    predictions = predictions[:len(df_test)]

# Here we avoid chained indexing by creating a new DataFrame for the test set and predictions.
df_test_with_predictions = pd.DataFrame({'Open': df_test['Open'], 'predictions': predictions})

# If there are any NaN values in 'Open' or 'predictions', we could fill them with the mean value of the respective columns
df_test_with_predictions['Open'].fillna((df_test_with_predictions['Open'].mean()), inplace=True)
df_test_with_predictions['predictions'].fillna((df_test_with_predictions['predictions'].mean()), inplace=True)

# Compute the RMSE
rmse = np.sqrt(mean_squared_error(df_test_with_predictions['Open'], df_test_with_predictions['predictions']))

print('Root Mean Squared Error:', rmse)
