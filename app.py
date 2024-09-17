import streamlit as st

import numpy as np
import cv2

from object_detection import predict

from components import footer

st.title("Object Detection")
st.text("This model (faster rcnn) trained on COCO dataset")

uploaded_file = st.file_uploader("Choose a image file", type="jpg")

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    image, text  = predict(opencv_image)

    st.image(image)
    st.markdown(f" #### {text}")

st.markdown(footer,unsafe_allow_html=True)