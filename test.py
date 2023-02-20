import numpy
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Assuming `num` is a list of 28000 data points
num = list(range(28000))  # Your data goes here

# Define the length of input sequences and the number of future predictions
sequence_length = 50
future_predictions = 1000

# Prepare the data for training
data = []
for i in range(len(num) - sequence_length):
    data.append(num[i:i+sequence_length+1])
data = np.array(data)

# Split the data into training and validation sets
split_ratio = 0.8
split_index = int(len(data) * split_ratio)
x_train = data[:split_index, :-1]
y_train = data[:split_index, -1]
x_val = data[split_index:, :-1]
y_val = data[split_index:, -1]

# Build the LSTM model
model = Sequential([
    LSTM(64, input_shape=(sequence_length, 1)),
    Dense(1)
])

model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(x_train.reshape((x_train.shape[0], x_train.shape[1], 1)), y_train,
          validation_data=(x_val.reshape((x_val.shape[0], x_val.shape[1], 1)), y_val),
          epochs=10, batch_size=64)

# Predict the future values
predictions = []
last_sequence = x_val[-1]
for i in range(future_predictions):
    prediction = model.predict(last_sequence.reshape((1, sequence_length, 1)))
    predictions.append(prediction[0, 0])
    last_sequence = np.concatenate((last_sequence[1:], prediction), axis=None)
predictions[0] = predictions[0] + data[len(data)-1]
for i in range(1,len(predictions)):
    predictions[i] = predictions[i] + predictions[i-1]
# Plot the predictions
import matplotlib.pyplot as plt

plt.plot(np.arange(len(num)), num, label='Actual')
plt.plot(np.arange(len(num), len(num) + future_predictions), predictions, label='Predicted')
plt.legend()
plt.show()
