import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Insulator Defect Detector")
st.write("Upload an insulator image to detect defects.")

model = YOLO('best.pt')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
   st.image(image, caption='Uploaded Image', use_container_width=True) 
    st.write("Detecting defects...")
    
    results = model.predict(image)
    result_img = results[0].plot()
    st.image(result_img, caption='Detection Result', use_column_width=True)
