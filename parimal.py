import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf




st.title("Hello Parimal")

def data_preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.resize(img, [1, 28, 28])
    img = img/255
    return img

image = st.file_uploader("Upload files", type=["jpeg", "png", "jpg", "webp"])
model = tf.keras.models.load_model("mnist.h5")

if image is not None:
    img = Image.open(image)
    img = np.array(img)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    images = data_preprocessing(img)
    predictions = model.predict(images)
    predictions = np.argmax(predictions)
    st.write(f"The Predicted Value: {predictions}")
    


    

