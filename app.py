import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Image Recognition",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

#MainMenu,
footer,
header{
    visibility:hidden;
}

.block-container{
    padding-top:1.5rem;
    padding-bottom:1rem;
    max-width:1100px;
}

div[data-testid="stMetric"]{
    border:1px solid #e5e7eb;
    border-radius:10px;
    padding:10px;
}

.stDataFrame{
    border-radius:10px;
}

hr{
    margin-top:0.5rem;
    margin-bottom:1rem;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess(image):
    """
    Preprocess image for MobileNetV2
    """

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    return preprocess_input(image_array)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict(image):
    """
    Predict top 5 ImageNet classes
    """

    batch = preprocess(image)

    start = time.perf_counter()

    predictions = model.predict(
        batch,
        verbose=0
    )

    inference_time = (
        time.perf_counter() - start
    ) * 1000

    results = decode_predictions(
        predictions,
        top=5
    )[0]

    return results, inference_time

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🖼️ Image Recognition")

    st.caption("MobileNetV2 • ImageNet")

    st.divider()

    st.markdown("### Model")

    st.write("**Architecture:** MobileNetV2")
    st.write("**Dataset:** ImageNet")
    st.write("**Classes:** 1000")

    st.divider()

    st.markdown("### Supported Formats")

    st.write("• JPG")
    st.write("• JPEG")
    st.write("• PNG")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🖼️ Image Recognition")

st.caption(
    "Upload an image and identify objects using MobileNetV2."
)

st.divider()


# --------------------------------------------------
# UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Upload a JPG, JPEG or PNG image to begin.")
    st.stop()

image = Image.open(uploaded_file)

file_size = len(uploaded_file.getvalue()) / 1024

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

with st.spinner("Analyzing image..."):
    results, inference_time = predict(image)

top_label = results[0][1].replace("_", " ").title()
top_confidence = results[0][2] * 100

# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------

left, right = st.columns([1, 1.1], gap="large")


with left:

    st.subheader("Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Width",
        image.width
    )

    c2.metric(
        "Height",
        image.height
    )

    c3.metric(
    "Size",
    f"{file_size:.1f} KB"
    )

    st.caption(f"Color Mode: {image.mode}")


with right:

    st.subheader("Prediction")

    st.metric(
        "Best Match",
        top_label,
        f"{top_confidence:.2f}%"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Confidence",
        f"{top_confidence:.2f}%"
    )

    c2.metric(
        "Prediction Time",
        f"{inference_time:.1f} ms"
    )

    st.markdown("### Top 5 Predictions")

    prediction_df = pd.DataFrame({
        "Rank": [1, 2, 3, 4, 5],
        "Prediction": [
            r[1].replace("_", " ").title()
            for r in results
        ],
        "Confidence (%)": [
            round(r[2] * 100, 2)
            for r in results
        ]
    })

    st.dataframe(
        prediction_df,
        hide_index=True,
        use_container_width=True
    )

st.write("")

csv = prediction_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Prediction Report",
    data=csv,
    file_name="prediction_report.csv",
    mime="text/csv",
    use_container_width=True,
)

st.divider()

st.caption(
    "Powered by TensorFlow • MobileNetV2"
)
