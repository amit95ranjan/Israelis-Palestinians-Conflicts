from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
X = data[['Palestinians Injuries', 'Israelis Injuries', 'Palestinians Killed', 'Israelis Killed']]
Y = data['Palestinians Injuries']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=32)
n_steps = 1
n_features = X.shape[1] # with one variable: Palestinians Killed

X_train_lstm = X_train.values.reshape(-1, n_steps, n_features)
X_test_lstm = X_test.values.reshape(-1, n_steps, n_features)

print(X_train_lstm.shape)
print(X_test_lstm.shape)
lstm_model = Sequential([
    LSTM(units=100, activation='relu', input_shape=(n_steps, n_features)),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, Y_train, epochs=50, batch_size=32)

rf_model = RandomForestRegressor()
rf_model.fit(X_train, Y_train)


y_pred_rf = rf_model.predict(X_test)
mae = mean_absolute_error(Y_test, y_pred_rf)
mse = mean_squared_error(Y_test, y_pred_rf)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae,     
     "\nMean Squared Error:", mse,
     "\nRoot Mean Squared Error:", rmse)
     start_date_future = data.index[-1] + pd.DateOffset(days=1)
end_date_future = start_date_future + pd.DateOffset(years=6)
print("Start Date for Future forecast :", start_date_future,
      "\nEnd date for Future forecast :", end_date_future)
      future_dates = pd.date_range(start=start_date_future, end=end_date_future, freq='Y')
X_future = pd.DataFrame(columns=X_train.columns, index=future_dates)
X_future.fillna(0, inplace=True)
future_forecast = rf_model.predict(X_future)
future_forecast
