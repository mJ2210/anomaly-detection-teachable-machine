import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()
class_names = ["Normal", "Anomaly"]

st.title("🧠 Anomaly Detection App")

# ----------------------------
# 📁 1. Image Upload Section
# ----------------------------
st.header("📷 Upload Image for Anomaly Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image')

    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"### 🔍 Prediction: **{predicted_class}**")

# ----------------------------
# 🎥 2. Live Camera Detection
# ----------------------------
st.markdown("---")
st.header("🎥 Real-Time Camera Anomaly Detection")

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Preprocess
        resized_img = cv2.resize(img, (224, 224))
        img_array = np.expand_dims(resized_img / 255.0, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        # Overlay prediction
        label = f"Prediction: {predicted_class}"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        return img

# Launch the webrtc video streamer
webrtc_streamer(
    key="realtime-anomaly-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
