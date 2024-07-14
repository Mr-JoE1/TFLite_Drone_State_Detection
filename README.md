# Drone State Detection Model

This repository contains a TensorFlow Lite model for detecting the state of a drone using IMU data. The model is designed to classify the drone's state into one of four categories: Stopped, Flying, Take off, and Landing.

## Model Description

The model is a neural network trained using TensorFlow/Keras. It takes as input nine features from the IMU (accelerometer, gyroscope, and magnetometer readings) and outputs one of four states. The model has been converted to TensorFlow Lite format for deployment on microcontrollers such as the ESP32.

### Features

- **Accelerometer Readings:** `Accel_X`, `Accel_Y`, `Accel_Z`
- **Gyroscope Readings:** `Gyro_X`, `Gyro_Y`, `Gyro_Z`
- **Magnetometer Readings:** `Mag_X`, `Mag_Y`, `Mag_Z`

### Labels

- **0:** Stopped
- **1:** Flying
- **2:** Take off
- **3:** Landing

## How to Use

### Prerequisites

- Arduino IDE or PlatformIO IDE
- ESP32 development board
- LSM9DS1 IMU sensor
- TensorFlow Lite for Microcontrollers library

### Build your own model using my dataset

1. **Clone the Repository:**
2. **cd to build_your_model folder**
3. **Follow step at How_To.md**
