import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
model_path = 'model.keras' 
model = tf.keras.models.load_model(model_path)

def preprocess_image(image):
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, dtype=tf.float32)

    image = image / 255.0

    return image

st.title("Superhero Machine Classification App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    processed_image = preprocess_image(np.array(image))

    prediction = model.predict(np.expand_dims(processed_image, 0))
    if prediction > 0.5:
        st.write("It's your friendly neighbourhood Spider-Man!")
    else:
        st.write("I am Batman.")
