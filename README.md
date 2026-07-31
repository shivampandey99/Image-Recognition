
# 🖼️ Image Classification using MobileNetV2

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red)
![License](https://img.shields.io/badge/License-MIT-green)

A modern **Image Classification** web application built with **TensorFlow**, **MobileNetV2**, and **Streamlit**. The application classifies uploaded images using a pretrained MobileNetV2 model trained on the ImageNet dataset and displays the Top-5 predictions with confidence scores.

---

## ✨ Features

- Upload JPG, JPEG, and PNG images
- Real-time image classification
- MobileNetV2 pretrained on ImageNet
- Top-5 predictions with confidence scores
- Image preview and metadata
- Fast inference
- Modern Streamlit interface
- Ready for Streamlit Community Cloud deployment

---

## 🧠 Model

| Property | Value |
|----------|-------|
| Model | MobileNetV2 |
| Framework | TensorFlow / Keras |
| Dataset | ImageNet |
| Classes | 1000 |
| Input Size | 224 × 224 |

This project uses **transfer learning** by leveraging a pretrained MobileNetV2 model. No additional model training is performed.

---

## ⚙️ Tech Stack

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow

---

## 📂 Project Structure

```text
Image-Recognition-Using-MobileNetV2/
│
├── app.py
├── Notebook.ipynb
├── README.md
├── requirements.txt
├── runtime.txt
├── .gitignore
└── assets/
```

---

## 🚀 Installation

```bash
git clone <repository-url>
cd Image-Recognition-Using-MobileNetV2
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📸 How It Works

1. Upload an image.
2. Convert the image to RGB.
3. Resize to **224 × 224**.
4. Apply MobileNetV2 preprocessing.
5. Run inference.
6. Decode ImageNet predictions.
7. Display the Top-5 predictions.

---

## 📷 Application Preview

Add a screenshot after deploying the application.

```text
assets/app_preview.png
```

---

## ⚠️ Limitations

- Supports only ImageNet classes.
- Not designed for object detection.
- Performance depends on image quality.

---

## 🔮 Future Improvements

- Fine-tune on a custom dataset
- Compare multiple CNN architectures
- Batch image classification
- Download prediction reports
- Webcam support

---

## 👨‍💻 Author

**Shivam Pandey**

B.Tech – Computer Science & Engineering (AI & ML)

---

## 📜 License

This project is released under the MIT License.
