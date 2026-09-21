import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

st.set_page_config(page_title="Insulator Defect Detector", layout="wide")

st.title("Intelligent Insulator Defect Detector")
st.write("Upload an insulator image to detect defects using Enhanced YOLOv8 + WBF.")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_h, img_w = img_bgr.shape[:2]

    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Detecting defects...")

    # 1. Predict with slightly higher confidence (0.25 instead of 0.1)
    results = model.predict(img_bgr, conf=0.25, iou=0.5, augment=True, half=True, verbose=False)

    # 2. Apply WBF
    boxes_list, scores_list, labels_list = [], [], []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        labels = r.boxes.cls.cpu().numpy()
        boxes[:, [0, 2]] /= img_w
        boxes[:, [1, 3]] /= img_h
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    # skip_box_thr=0.4 to ignore weak boxes
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, iou_thr=0.5, skip_box_thr=0.4
    )

    # 3. Draw final result with higher final filter (score > 0.5)
    found_defect = False
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        x1, y1, x2, y2 = box
        x1, x2 = int(x1 * img_w), int(x2 * img_w)
        y1, y2 = int(y1 * img_h), int(y2 * img_h)
        if score > 0.5:  # Only show if confidence is > 50%
            found_defect = True
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = f"defect {score:.2f}"
            cv2.putText(img_bgr, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if not found_defect:
        st.success("✅ No defects detected. This is a Good Insulator.")
    else:
        st.warning("⚠️ Defects detected in the image!")

    # Convert BGR back to RGB for Streamlit display
    result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption='Detection Result', use_container_width=True)