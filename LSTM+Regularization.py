from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Post-Sterilization CFU']].values)

def create_sequences(data, window_size):
    sequences, targets = [], []
    for i in range(window_size, len(data)):
        sequences.append(data[i-window_size:i, 0])
        targets.append(data[i, 0])
    return np.array(sequences), np.array(targets)

window_size = 10
X, y = create_sequences(scaled_data, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), kernel_regularizer='l2'))
model.add(Dropout(0.3))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train Loss', 'Validation Loss'])
plt.title('Training and Validation Loss using Adam')
plt.show()

y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)

mse = mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), y_pred_rescaled)
print(f"Mean Squared Error: {mse}")