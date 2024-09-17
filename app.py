import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import google.generativeai as genai
import os
  

# Load your model (change 'your_model.h5' to your actual model file)
model = tf.keras.models.load_model('traffic_efficient_net_B0_v1.keras')
# Correctly access the secret
api_key = st.secrets["my_secret"]

labels_df = pd.read_csv('labels.csv')

genai.configure(api_key=api_key)
# Define the image size expected by your model
IMG_SIZE = (224, 224)  # Update this according to your model's input size

# Define class names based on your model
class_names = np.array([1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5, 50, 51, 6, 7, 8, 9])


def preprocess_image(img):
    # Convert image to RGB
    img = img.convert('RGB')
    # Resize image to the expected input shape
    img = img.resize((IMG_SIZE))
    # Convert image to numpy array
    img_array = np.array(img)
    # Scale the image
    img_array = img_array
    # Expand dimensions to fit model input
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_prediction(img):
    # Preprocess the image
    img_array = preprocess_image(img)
    # Make prediction
    preds = model.predict(img_array)
    # Get the class index with the highest probability
    predicted_class_index = np.argmax(preds[0])
    predicted_class_name = labels_df.iloc[int(class_names[int(predicted_class_index)])].Name
    return predicted_class_name

# Streamlit application
st.title('Traffic Light Classifier with EfficientNetV2-B0')
st.write('This model is a deep learning model specifically trained to recognize and classify over 44 different types of traffic lights from images. It leverages the EfficientNetV2-B0 architecture, a pre-trained convolutional neural network known for its efficiency and accuracy. The model is fine-tuned on a large-scale traffic light image dataset, allowing it to learn specific features and patterns associated with various traffic light types.')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    prediction = make_prediction(image)
    google = genai.GenerativeModel('gemini-1.5-flash')
    response = google.generate_content(f'In about 40 words tell me something fun about {prediction}s road sign.')
    
    # Display prediction
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Fun Fact: 📖** {response.text}")

    st.write("")
    st.write("By RickmwasOfficial 2024")