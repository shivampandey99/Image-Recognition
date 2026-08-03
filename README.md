# 🖼️ Image Recognition

A clean, minimal Streamlit application for image classification using **MobileNetV2** pretrained on **ImageNet**.

## Features

- Clean and minimal Streamlit interface
- Upload JPG, JPEG, and PNG images
- AI-powered image classification using MobileNetV2
- Displays the best prediction with confidence score
- Top 5 predicted classes
- Image details (Width, Height, Size, Color Mode)
- Prediction time measurement
- Download prediction report as CSV
- Ready for GitHub and Streamlit Cloud deployment

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
Image-Recognition/
│
├── assets/
│   ├── sample_dog.jpg
│   └── sample_flower.jpg
│
├── app.py
├── Image Recognition Notebook.ipynb
├── requirements.txt
├── runtime.txt
└── README.md
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


---

## 👨‍💻 Author

**Shivam Pandey**

B.Tech — Computer Science & Engineering  
Specialization: Artificial Intelligence & Machine Learning