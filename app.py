import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    WebRtcMode,
    ClientSettings,
)

# ----------------------------
# Load the model once
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()
class_names = ["Normal", "Anomaly"]

# ----------------------------
# Page UI
# ----------------------------
st.set_page_config(page_title="Anomaly Detector", layout="centered")
st.title("🧠 Anomaly Detection")
st.markdown("Choose a mode below to test for anomalies:")

mode = st.radio("Select Mode", ["📤 Upload Image", "📷 Camera Snapshot", "🎥 Live Camera (Local Only)"])

# ----------------------------
# 📤 Upload Image
# ----------------------------
if mode == "📤 Upload Image":
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")

        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        st.success(f"🔍 Prediction: **{predicted_class}**")

# ----------------------------
# 📷 Camera Snapshot (Streamlit Cloud Friendly)
# ----------------------------
elif mode == "📷 Camera Snapshot":
    st.header("Take a Picture")
    image_data = st.camera_input("Capture using your webcam")
    if image_data is not None:
        image = Image.open(image_data).convert("RGB")
        st.image(image, caption="Captured Image")

        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        st.success(f"🔍 Prediction: **{predicted_class}**")

# ----------------------------
# 🎥 Live Camera (Only for Local or Custom Cloud)
# ----------------------------
elif mode == "🎥 Live Camera (Local Only)":
    st.header("Real-Time Anomaly Detection")

    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            resized_img = cv2.resize(img, (224, 224))
            img_array = np.expand_dims(resized_img / 255.0, axis=0)

            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]

            label = f"Prediction: {predicted_class}"
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            return img

    # Changing the key can refresh broken WebRTC sessions
    webrtc_streamer(
        key="live-camera-v2",  # 🔁 change key if camera isn't rendering
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        client_settings=ClientSettings(
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
    )
