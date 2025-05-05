import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="YOLOv8 Car Detection", layout="centered")



st.title("🚗 Car Detection using YOLOv8")

uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    return YOLO("D:/DS-Class/Sep_Proj/Pro_1/yolov8n.pt")

model = load_model()

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        image_np = np.array(image)

        results = model.predict(
            source=image_np,
            verbose=False
        )


        result_img = results[0].plot()

        st.image(result_img, caption="✅ Detection Result", use_container_width=True)

        with st.expander("📋 Detection Summary"):
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = float(box.conf)
                st.markdown(f"- **Class**: {class_name} | **Confidence**: `{confidence:.2f}`")

    except Exception as e:
        st.error(f"❌ Error processing the image: {e}")
