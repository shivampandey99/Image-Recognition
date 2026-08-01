
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

st.set_page_config(page_title="Image Recognition", page_icon="🖼️", layout="wide")

st.markdown("""
<style>
#MainMenu,header,footer{visibility:hidden}
.block-container{max-width:1050px;padding-top:1.2rem;padding-bottom:1rem}
html,body,[data-testid="stAppViewContainer"]{background:#fafafa}
h1,h2,h3{font-weight:600}
hr{margin:.7rem 0}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #eee}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model=load_model()

with st.sidebar:
    st.title("Image Recognition")
    st.caption("MobileNetV2 • ImageNet")
    st.divider()
    st.markdown("**Features**")
    st.markdown("- Upload JPG/PNG\n- Top‑5 predictions\n- CSV export\n- Fast inference")

def preprocess(img):
    img=img.convert("RGB").resize((224,224))
    arr=tf.keras.utils.img_to_array(img)
    arr=np.expand_dims(arr,0)
    return preprocess_input(arr)

def predict(img):
    batch=preprocess(img)
    t=time.perf_counter()
    pred=model.predict(batch,verbose=0)
    ms=(time.perf_counter()-t)*1000
    return decode_predictions(pred,top=5)[0],ms

st.title("Image Recognition")
st.caption("Upload an image and classify it using MobileNetV2.")

f=st.file_uploader("Choose an image",type=["jpg","jpeg","png"])
if not f:
    st.info("Upload an image to begin.")
    st.stop()

img=Image.open(f)
results,ms=predict(img)

c1,c2=st.columns([1,1.1],gap="large")
with c1:
    st.image(img,use_container_width=True)
    a,b,c=st.columns(3)
    a.metric("Width",img.width)
    b.metric("Height",img.height)
    c.metric("Size",f"{len(f.getvalue())/1024:.1f} KB")

with c2:
    top=results[0]
    st.metric("Prediction",top[1].replace("_"," ").title(),f"{top[2]*100:.2f}%")
    st.metric("Inference",f"{ms:.1f} ms")
    df=pd.DataFrame({
        "Rank":range(1,6),
        "Prediction":[r[1].replace("_"," ").title() for r in results],
        "Confidence (%)":[round(r[2]*100,2) for r in results]
    })
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.download_button("Download CSV",df.to_csv(index=False).encode(),"prediction_report.csv","text/csv")
    for _,label,p in results:
        st.write(label.replace("_"," ").title())
        st.progress(float(p))

st.caption("Powered by TensorFlow • MobileNetV2")
