
# Drone State Detection Model

This guide provides step-by-step instructions for loading, preprocessing, visualizing IMU data, training a TensorFlow model, converting it to TensorFlow Lite, and deploying it to an ESP32 using the Arduino PlatformIO IDE.

## Step 1: Install Required Libraries

First, install the necessary Python libraries:

```sh
pip install pandas matplotlib seaborn scikit-learn tensorflow
```

## Step 2: Load and Preprocess the Data

### 2.1 Load the Data

Load the `drone_imu_data.csv` file into a Pandas DataFrame, handling any inconsistent rows:

```python
import pandas as pd

try:
    # Load the data and skip bad lines
    data = pd.read_csv('drone_imu_data.csv', on_bad_lines='skip')
    print("Data loaded successfully")
except FileNotFoundError:
    print("The file 'drone_imu_data.csv' was not found.")
    exit(1)

# Display the first few rows of the dataframe
print(data.head())
```

### 2.2 Preprocess the Data

Encode the labels, normalize the features, and split the data into training and testing sets:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Check for missing values and drop rows with missing values if any
data.dropna(inplace=True)

# Separate features and labels
X = data[['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Mag_X', 'Mag_Y', 'Mag_Z']]
y = data['State']

# Encode the state labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Combine the scaled features and labels into a single DataFrame for visualization
data_scaled = pd.DataFrame(X_scaled, columns=['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Mag_X', 'Mag_Y', 'Mag_Z'])
data_scaled['State'] = y
```

## Step 3: Visualize the Data

### 3.1 Pairplot

Use Seaborn’s `pairplot` to visualize pairwise relationships in the dataset:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Convert the state labels back to their original string labels for visualization
data_scaled['State'] = label_encoder.inverse_transform(data_scaled['State'])

# Create a pairplot
sns.pairplot(data_scaled, hue='State', diag_kind='kde', markers=["o", "s", "D", "X"])
plt.suptitle("Pairplot of IMU Data", y=1.02)
plt.show()
```

### 3.2 Correlation Heatmap

Visualize the correlation between different features using a heatmap:

```python
# Calculate the correlation matrix
corr_matrix = data_scaled.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of IMU Data")
plt.show()
```

### 3.3 Boxplot

Use boxplots to visualize the distribution of each feature across different states:

```python
# Create boxplots for each feature
plt.figure(figsize=(15, 10))
for i, column in enumerate(data_scaled.columns[:-1]):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='State', y=column, data=data_scaled)
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()
```

### 3.4 Time Series Plot

If you have a time column, plot the time series for a better understanding of how the IMU readings change over time:

```python
# Assuming there's a 'Time' column in the original data
data['Time'] = pd.to_datetime(data['Time'])

plt.figure(figsize=(15, 10))
for i, column in enumerate(['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Mag_X', 'Mag_Y', 'Mag_Z']):
    plt.subplot(3, 3, i+1)
    sns.lineplot(x='Time', y=column, hue='State', data=data)
    plt.title(f'Time Series of {column}')
plt.tight_layout()
plt.show()
```

## Step 4: Define and Train the Model

### 4.1 Define the Model

Create a neural network model using TensorFlow/Keras:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')  # Assuming 4 classes: Stopped, Flying, Take off, Landing
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### 4.2 Train the Model

Train the model using the training data:

```python
# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')
```

## Step 5: Convert the Model to TensorFlow Lite

### 5.1 Convert the Model to TensorFlow Lite Format

Convert the trained model to TensorFlow Lite format for deployment on microcontrollers:

```python
# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to a file
with open('drone_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 5.2 Convert the Model to a Byte Array

Convert the `.tflite` file to a C-style byte array and create the `drone_model_data.h` file:

```python
import numpy as np

# Load your TensorFlow Lite model
model_path = 'drone_model.tflite'
with open(model_path, 'rb') as f:
    model_content = f.read()

# Convert the model content to a byte array
byte_array = np.frombuffer(model_content, dtype=np.uint8)

# Generate the C header file content
header_content = f"""
#ifndef DRONE_MODEL_DATA_H_
#define DRONE_MODEL_DATA_H_

extern const unsigned char g_drone_model_data[];
extern const int g_drone_model_data_len;

const unsigned char g_drone_model_data[] = {{
    {', '.join(f'0x{byte:02x}' for byte in byte_array)}
}};
const int g_drone_model_data_len = {len(byte_array)};

#endif  // DRONE_MODEL_DATA_H_
"""

# Write the header content to a file
with open('drone_model_data.h', 'w') as f:
    f.write(header_content)

print('drone_model_data.h file created successfully.')
```

## Step 6: Deploy the Model to ESP32 Using PlatformIO

### 6.1 Create a New PlatformIO Project

- Open PlatformIO IDE and create a new project for the ESP32.
- Copy the `drone_model_data.h` file to the `src` directory of your PlatformIO project.
- Update the `platformio.ini` file to include the TensorFlow Lite library.

```ini
[env:esp32]
platform = espressif32
board = esp32dev
framework = arduino
lib_deps = 
    https://github.com/tensorflow/tflite-micro-arduino-examples
```

## Step 7: Upload and Monitor

1. **Upload the Code:**
   - Connect your ESP32 board to your computer.
   - Upload the code to the ESP32 using PlatformIO.

2. **Monitor the Output:**
   - Open the Serial Monitor in PlatformIO to see the detected drone states in real-time.

## Summary

This guide covers the following steps to train and deploy a drone state detection model:

1. **Install Required Libraries:** Ensure all necessary libraries are installed.
2. **Load and Preprocess the Data:** Load the dataset, handle inconsistencies, encode labels, and normalize features.
3. **Visualize the Data:** Use pairplots, correlation heatmaps, boxplots, and time series plots to visualize the dataset.
4. **Define and Train the Model:** Create and train a neural network model using TensorFlow/Keras.
5. **Convert the Model to TensorFlow Lite:** Convert the trained model to TensorFlow Lite format for deployment on microcontrollers.
6. **Deploy the Model to ESP32 Using PlatformIO:** Set up a PlatformIO project, include the TensorFlow Lite library, and write the Arduino sketch to run inference on the ESP32.
