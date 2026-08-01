import time
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
)

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="AI Image Recognition",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.block-container{
padding-top:2rem;
padding-bottom:1rem;
max-width:1200px;
}

/* ---------- Hero ---------- */

.hero{
background:linear-gradient(135deg,#2563eb,#7c3aed);
padding:30px;
border-radius:18px;
color:white;
margin-bottom:25px;
box-shadow:0 8px 20px rgba(0,0,0,.18);
}

.hero h1{
font-size:42px;
margin-bottom:8px;
}

.hero p{
font-size:18px;
opacity:.92;
}

/* ---------- Cards ---------- */

.card{
background:white;
padding:22px;
border-radius:16px;
box-shadow:0 4px 14px rgba(0,0,0,.08);
border:1px solid #e8eef7;
margin-bottom:20px;
}

.metric-card{
background:linear-gradient(135deg,#ffffff,#f5f7ff);
padding:18px;
border-radius:16px;
text-align:center;
box-shadow:0 2px 10px rgba(0,0,0,.06);
border:1px solid #e9eef8;
}

.metric-title{
font-size:15px;
color:#6b7280;
}

.metric-value{
font-size:28px;
font-weight:700;
color:#2563eb;
}

/* ---------- Prediction ---------- */

.prediction{
background:linear-gradient(135deg,#22c55e,#16a34a);
padding:18px;
border-radius:15px;
color:white;
text-align:center;
margin-bottom:15px;
}

.prediction h2{
margin:0;
}

.prediction p{
margin:5px 0;
font-size:18px;
}

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"]{
background:#0f172a;
}

section[data-testid="stSidebar"] *{
color:white !important;
}

/* ---------- Upload ---------- */

.upload-box{
border:2px dashed #2563eb;
padding:20px;
border-radius:15px;
background:#f8fbff;
text-align:center;
margin-bottom:15px;
}

/* ---------- Footer ---------- */

.footer{
margin-top:40px;
text-align:center;
padding:20px;
color:#6b7280;
font-size:15px;
border-top:1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------

@st.cache_resource
def load_model():

    with st.spinner("Loading MobileNetV2 Model..."):

        return MobileNetV2(weights="imagenet")

model = load_model()

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

with st.sidebar:

    st.image(
        "https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/tensorflow.svg",
        width=80,
    )

    st.title("AI Image Recognition")

    st.markdown("---")

    st.subheader("📌 Project Information")

    st.info(
        """
**Model**
MobileNetV2

**Dataset**
ImageNet

**Framework**
TensorFlow / Keras

**Input Size**
224 × 224

**Output Classes**
1000
"""
    )

    st.markdown("---")

    st.subheader("✨ Features")

    st.success("✔ Upload JPG / PNG images")
    st.success("✔ AI-based image recognition")
    st.success("✔ Top 5 predictions")
    st.success("✔ Confidence scores")
    st.success("✔ Fast inference")

    st.markdown("---")

    st.subheader("👨‍💻 Developer")

    st.markdown(
        """
**Shivam Pandey**

B.Tech — Computer Science & Engineering

Specialization:
Artificial Intelligence & Machine Learning
"""
    )

# -----------------------------------------------------
# IMAGE PREPROCESSING
# -----------------------------------------------------

def preprocess(image):

    image = image.convert("RGB")

    image = image.resize((224,224))

    image_array = tf.keras.utils.img_to_array(image)

    image_array = np.expand_dims(image_array, axis=0)

    return preprocess_input(image_array)

# -----------------------------------------------------
# PREDICTION
# -----------------------------------------------------

def predict(image):

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

# -----------------------------------------------------
# UPLOAD SECTION
# -----------------------------------------------------

st.markdown(
"""
<div class="upload-box">

<h3>📤 Upload an Image</h3>

Supported Formats

<b>JPG • JPEG • PNG</b>

</div>
""",
unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "",
    type=["jpg","jpeg","png"]
)

# -----------------------------------------------------
# NO IMAGE
# -----------------------------------------------------

if uploaded_file is None:

    st.info(
        "👆 Upload an image to begin AI image recognition."
    )

    st.stop()

# -----------------------------------------------------
# LOAD IMAGE
# -----------------------------------------------------

image = Image.open(uploaded_file)

file_size = len(uploaded_file.getvalue()) / 1024

left,right = st.columns([1,1])

# -----------------------------------------------------
# LEFT PANEL
# -----------------------------------------------------

with left:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🖼 Uploaded Image")

    st.image(
    image,
    use_column_width=True
    )

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Width",
        image.width
    )

    col2.metric(
        "Height",
        image.height
    )

    col3.metric(
        "Size",
        f"{file_size:.1f} KB"
    )

    st.write("**Color Mode:**", image.mode)

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

# -----------------------------------------------------
# PREDICT
# -----------------------------------------------------

with st.spinner("🤖 Analyzing image..."):

    results, inference_time = predict(image)


# -----------------------------------------------------
# RIGHT PANEL
# -----------------------------------------------------

with right:

    top_label = results[0][1].replace("_", " ").title()
    top_confidence = results[0][2] * 100

    st.markdown(
        f"""
<div class="prediction">
<h2>🏆 Best Prediction</h2>
<p style="font-size:28px;font-weight:bold;">{top_label}</p>
<p>Confidence: {top_confidence:.2f}%</p>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
<div class="metric-card">
<div class="metric-title">Confidence</div>
<div class="metric-value">{top_confidence:.2f}%</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
<div class="metric-card">
<div class="metric-title">Inference</div>
<div class="metric-value">{inference_time:.1f} ms</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
<div class="metric-card">
<div class="metric-title">Classes</div>
<div class="metric-value">1000</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("📊 Top 5 Predictions")

    colors = [
        "🥇",
        "🥈",
        "🥉",
        "4️⃣",
        "5️⃣"
    ]

    report = []

    for i, (_, label, prob) in enumerate(results):

        label = label.replace("_", " ").title()

        percent = prob * 100

        report.append(
            {
                "Rank": i + 1,
                "Prediction": label,
                "Confidence (%)": round(percent, 2),
            }
        )

        prediction_df = pd.DataFrame({
    "Rank": ["🥇","🥈","🥉","4️⃣","5️⃣"],
    "Prediction": [
        r[1].replace("_"," ").title()
        for r in results
    ],
    "Confidence (%)": [
        round(r[2]*100,2)
        for r in results
    ]
})

st.dataframe(
    prediction_df,
    use_container_width=True,
    hide_index=True
)


st.progress(float(prob))

st.caption(f"{percent:.2f}%")

# -----------------------------------------------------
# DOWNLOAD REPORT
# -----------------------------------------------------

import pandas as pd

df = pd.DataFrame(report)

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download Prediction Report",
    csv,
    file_name="prediction_report.csv",
    mime="text/csv",
    use_container_width=True,
)

# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------
st.markdown("---")

st.caption(
    "Powered by TensorFlow • MobileNetV2"
)