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
