import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()
class_names = ["Normal", "Anomaly"]

# App UI
st.title("🔍 Anomaly Detection App")
st.markdown("Upload an image to check for anomalies.")

# Image upload prediction
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image')

        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"### 🧠 Prediction: **{predicted_class}** ({confidence:.1f}%)")
    except Exception as e:
        st.error(f"❌ Failed to process image: {e}")
else:
    st.info("Please upload an image.")

# Divider
st.markdown("---")
st.header("📷 Real-Time Anomaly Detection")

# Streamlit WebRTC camera support
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        resized = cv2.resize(img, (224, 224))
        input_img = np.expand_dims(resized / 255.0, axis=0)

        prediction = model.predict(input_img)
        anomaly_score = prediction[0][1]  # Index 1 is assumed to be "Anomaly"
        threshold = 0.5  # You can adjust this

        predicted_class = "Anomaly" if anomaly_score >= threshold else "Normal"

        # Draw label
        label = f"{predicted_class}"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

        # Optional: show anomaly score
        cv2.putText(img, f"Anomaly score: {anomaly_score:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Activate webcam in browser
webrtc_streamer(
    key="anomaly-stream",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
