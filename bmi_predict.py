# bmi_predict.py
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Function to load the BMI prediction model with the provided file path
def load_bmi_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to preprocess an image and predict BMI
def predict_bmi_from_image(model, image_bytes, target_size):
    try:
        # Load the image from bytes and preprocess it
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize(target_size)  # Resize the image to the specified target size
        image_array = np.array(image) / 255.0  # Normalize pixel values

        # Add batch dimension for model input
        image_array = np.expand_dims(image_array, axis=0)

        # Predict BMI using the model
        predicted_bmi = model.predict(image_array)
        return predicted_bmi[0][0]
    except Exception as e:
        print(f"Error processing image and predicting BMI: {e}")
        return None
