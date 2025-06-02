import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os

# --- THIS IS THE CHANGE: Move st.set_page_config to the very top ---
st.set_page_config(page_title="Pneumonia X-Ray Classifier", layout="centered")

# --- Configuration ---
IMG_HEIGHT = 500
IMG_WIDTH = 500
CLASS_LABELS = {0: 'NORMAL', 1: 'PNEUMONIA'}
MODEL_PATH = 'medical_image_classifier_model.h5'

# --- Model Loading (using st.cache_resource for efficiency) ---
@st.cache_resource
def load_my_model():
    """
    Loads the pre-trained Keras model.
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at '{MODEL_PATH}'. "
                 "Please ensure the model directory/file is in the same directory as app.py.")
        return None
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the model (this will now run AFTER set_page_config)
model = load_my_model()

# --- Image Preprocessing Function ---
def preprocess_image(image):
    """
    Preprocesses the uploaded image to match the model's input requirements.
    """
    # ... (rest of your preprocess_image function remains the same)
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    if image.mode != 'L':
        image = image.convert('L')
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

# --- Streamlit Application UI (main function) ---
def main():
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.5em;
            color: #2E86C1;
            text-align: center;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-size: 1.1em;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stFileUploader {
            border: 2px dashed #ddd;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stImage {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .prediction-box {
            background-color: #EBF5FB;
            border-left: 5px solid #2E86C1;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .confidence-text {
            font-size: 1.1em;
            color: #555;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 class='main-header'>Pneumonia X-Ray Classifier</h1>", unsafe_allow_html=True)
    st.write("Upload a chest X-ray image (JPG, JPEG, PNG) to get a prediction "
             "on whether it shows signs of pneumonia or is normal.")

    if model is None:
        st.warning("Model could not be loaded. Please check the console for errors.")
        return

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded X-Ray Image.', use_column_width=True, width=300)
        st.write("")

        if st.button("Classify Image"):
            with st.spinner("Classifying image..."):
                processed_img = preprocess_image(image)
                predictions = model.predict(processed_img)
                predicted_class_index = np.argmax(predictions)
                confidence = predictions[0][predicted_class_index]
                predicted_label = CLASS_LABELS.get(predicted_class_index, "Unknown")

                st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                st.subheader(f"Prediction: **{predicted_label}**")
                st.markdown(f"<p class='confidence-text'>Confidence: {confidence:.2f}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.write("---")
                st.subheader("All Class Probabilities:")
                for idx, prob in enumerate(predictions[0]):
                    label = CLASS_LABELS.get(idx, f"Class {idx}")
                    st.write(f"{label}: {prob:.4f}")

if __name__ == "__main__":
    main()