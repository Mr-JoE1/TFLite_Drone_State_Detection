import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


##  Load and Preprocess the Data
try:
    # Load the data and skip bad lines
    data = pd.read_csv('drone_imu_data.csv', on_bad_lines='skip')
    print("Data loaded successfully")
except FileNotFoundError:
    print("The file 'drone_imu_data.csv' was not found.")
    exit(1)

# Display the first few rows of the dataframe
print(data.head())

## Preprocess the Data
# Check for missing values and drop rows with missing values if any
data.dropna(inplace=True)

# Separate features and labels
X = data[['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Mag_X', 'Mag_Y', 'Mag_Z']]
y = data['State']

# Encode the state labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#print(X_train)
#print(X_test)

## Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')  # Assuming 4 classes: Stopped, Flying, Take off, Landing
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


## Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')


## Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to a file
with open('drone_model.tflite', 'wb') as f:
    f.write(tflite_model)
