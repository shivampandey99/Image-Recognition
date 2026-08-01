# 🖼️ Image Recognition

A clean, minimal Streamlit application for image classification using **MobileNetV2** pretrained on **ImageNet**.

## Features

- Minimal Apple-inspired interface
- Upload JPG, JPEG, or PNG
- Top-5 predictions with confidence
- Inference time
- Download predictions as CSV
- Streamlit & GitHub ready

## Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- MobileNetV2
- Pillow
- NumPy
- Pandas

## Project Structure

```text
image-recognition/
├── app.py
├── Image Recognition Notebook.ipynb
├── requirements.txt
├── README.md
└── assets/
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model

- MobileNetV2
- Input: 224×224
- Dataset: ImageNet
- Classes: 1000
