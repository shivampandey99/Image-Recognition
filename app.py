
import time
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

st.set_page_config(page_title="Image Classification using MobileNetV2", page_icon="🖼️", layout="wide")

st.markdown("""
<style>
.title{font-size:42px;font-weight:700;color:#2E86DE;}
.subtitle{font-size:18px;color:#666;}
.card{padding:18px;border-radius:12px;background:#f7f9fc;border:1px solid #ddd;margin-bottom:12px;}
.footer{text-align:center;color:gray;padding-top:25px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with st.spinner("Loading model..."):
        return MobileNetV2(weights="imagenet")

model = load_model()

def preprocess(image):
    img=image.convert("RGB").resize((224,224))
    arr=tf.keras.utils.img_to_array(img)
    arr=np.expand_dims(arr,0)
    return preprocess_input(arr)

def predict(image):
    batch=preprocess(image)
    start=time.perf_counter()
    preds=model.predict(batch,verbose=0)
    elapsed=(time.perf_counter()-start)*1000
    return decode_predictions(preds,top=5)[0],elapsed

with st.sidebar:
    st.header("About")
    st.write("Model: MobileNetV2\n\nDataset: ImageNet\n\nClasses:1000")

st.markdown('<div class="title">🖼️ Image Classification using MobileNetV2</div>',unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and classify it using a pretrained CNN.</div>',unsafe_allow_html=True)

uploaded=st.file_uploader("Upload Image",type=["jpg","jpeg","png"])

if uploaded:
    image=Image.open(uploaded)
    c1,c2=st.columns(2)
    with c1:
        st.image(image, use_column_width=True)
        st.write(f"Dimensions: {image.width} x {image.height}")
        st.write(f"Mode: {image.mode}")
    results,ms=predict(image)
    with c2:
        st.subheader("Top Prediction")
        st.success(results[0][1].replace("_"," ").title())
        st.metric("Confidence",f"{results[0][2]*100:.2f}%")
        st.metric("Inference",f"{ms:.1f} ms")
        st.subheader("Top 5")
        for _,label,p in results:
            st.write(label.replace("_"," ").title())
            st.progress(float(p))
            st.caption(f"{p*100:.2f}%")
        with st.expander("Technical Details"):
            st.write("- MobileNetV2\n- ImageNet\n- 224x224 Input")
else:
    st.info("Upload an image to begin.")

st.markdown('<div class="footer">Developed with TensorFlow, Streamlit & MobileNetV2</div>',unsafe_allow_html=True)
