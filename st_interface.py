import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import requests
import os

# Set Streamlit page config
st.set_page_config(page_title="ASL Gesture Classifier", layout="centered")

# Dropbox direct download link (replace this with your actual link)
dropbox_url = "https://www.dropbox.com/scl/fi/nmktcdiralyyqs8yld9jg/asl_gesture_model.keras?rlkey=c9jdf65bhpqlxfrc61ndo2k44&st=211jzstc&dl=1"
model_filename = "asl_gesture_model.keras"

# Download model if not already present
if not os.path.exists(model_filename):
    with st.spinner("Downloading model..."):
        r = requests.get(dropbox_url)
        with open(model_filename, "wb") as f:
            f.write(r.content)
    st.success("Model downloaded successfully!")

# Load the model
model = tf.keras.models.load_model(model_filename)
st.success("Model loaded successfully!")

# Class labels (29)
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                'del', 'nothing', 'space']

# Title
st.title("🤟 ASL Gesture Image Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image of a hand showing an ASL gesture", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess image
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)[0]
    predicted_idx = np.argmax(prediction)
    predicted_label = class_labels[predicted_idx]
    confidence = prediction[predicted_idx] * 100

    # Show prediction result
    st.markdown(f"### 🧠 Predicted Gesture: `{predicted_label}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")

    # Show top 5 predictions in a table
    top_indices = prediction.argsort()[-5:][::-1]
    top_labels = [class_labels[i] for i in top_indices]
    top_confidences = [prediction[i] * 100 for i in top_indices]

    df = pd.DataFrame({
        'Class': top_labels,
        'Confidence (%)': [f"{conf:.2f}" for conf in top_confidences]
    })

    st.markdown("### 🔍 Top 5 Predictions")
    st.table(df)
