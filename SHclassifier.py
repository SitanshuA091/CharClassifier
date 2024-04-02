import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import  model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

def preprocess_image(image):
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, dtype=tf.float32)

    image = image / 255.0

    return image

st.title("Superhero Machine Classification App")
Img = Image.open('SH.jpg')
st.sidebar.info('This binary Classifier classifies the identity of the Superhero (either Spider-Man or Batman')
st.sidebar.image(Img)


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
