import tensorflow as tf
import os

keras_model_path = "best_model.keras"
tflite_model_path = "best_model.tflite"

print(f"Loading Keras model from {keras_model_path}...")
model = tf.keras.models.load_model(keras_model_path)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Success! TFLite model saved to {tflite_model_path}")
print(f"Size: {os.path.getsize(tflite_model_path) / 1024:.2f} KB")
