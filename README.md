# 🖼️ AI Image Recognition using MobileNetV2

An AI-powered Image Recognition web application that I developed using **TensorFlow**, **MobileNetV2**, and **Streamlit**. This application allows users to upload an image and instantly identify the object present in the image using a pretrained Deep Learning model.

To make the project more user-friendly, I designed a modern Streamlit interface featuring a responsive dashboard, image preview, prediction cards, confidence visualization, downloadable prediction reports, and session history while keeping the underlying Deep Learning pipeline efficient and lightweight.

---

# 🚀 Live Features

- 📤 Upload JPG, JPEG and PNG images
- 🖼️ Live image preview
- 🤖 AI-powered object recognition
- 🏆 Displays the best prediction
- 📊 Top 5 predicted classes
- 📈 Confidence visualization using progress bars
- ⚡ Inference time measurement
- 📐 Displays uploaded image information
- 📥 Download prediction report (CSV)
- 🕘 Prediction history during the session
- 🎨 Modern responsive Streamlit UI
- 🌙 Professional dashboard layout
- ☁️ Ready for Streamlit deployment

---

# 📷 Application Preview

```
                AI Image Recognition

Upload Image
        │
        ▼
Image Preview + Image Information
        │
        ▼
MobileNetV2 Prediction
        │
        ▼
Top Prediction
        │
        ▼
Top 5 Predictions
        │
        ▼
Download Prediction Report
```

---

# 🎯 Project Objective

The objective of this project is to implement an end-to-end Image Recognition system using a pretrained Convolutional Neural Network.

Instead of training a Deep Learning model from scratch, I used **MobileNetV2 pretrained on ImageNet** to perform image classification efficiently while focusing on building a production-ready AI web application.

---

# 🧠 Model Used

## MobileNetV2

For image classification, I use the pretrained **MobileNetV2** architecture available in TensorFlow/Keras.

```python
MobileNetV2(weights="imagenet")
```

The model has already been trained on the **ImageNet** dataset containing more than one million images across **1000 object categories**.

Therefore, the application can recognize a wide variety of real-world objects without requiring additional model training.

---

# ⚙️ Technologies Used

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras
- MobileNetV2

### Image Processing

- Pillow (PIL)
- NumPy

### Web Application

- Streamlit

### Development Environment

- Google Colab
- VS Code

### Version Control

- Git
- GitHub

---

# 🏗 Project Architecture

```text
                    USER
                      │
                      ▼
              Upload Image
                      │
                      ▼
             Image Preprocessing
                      │
                      ▼
              Resize 224 × 224
                      │
                      ▼
         MobileNetV2 Preprocessing
                      │
                      ▼
         Pretrained MobileNetV2
                      │
                      ▼
         ImageNet Classification
                      │
                      ▼
      Decode Top 5 Predictions
                      │
                      ▼
      Streamlit Dashboard Output
```

---

# 🔍 Image Processing Pipeline

The uploaded image follows the processing pipeline shown below.

```
Upload Image

↓

Convert to RGB

↓

Resize (224 × 224)

↓

Convert to NumPy Array

↓

Expand Dimensions

↓

preprocess_input()

↓

MobileNetV2

↓

decode_predictions()

↓

Top 5 Predictions
```

---

# 🎨 User Interface

I redesigned the application with a modern Streamlit interface to improve usability and user experience.

The interface includes:

- Gradient hero banner
- Professional sidebar
- Responsive layout
- Upload section
- Image preview card
- Image information panel
- Best prediction card
- Confidence metrics
- Top 5 prediction cards
- Progress bars
- Prediction history
- Download prediction report
- Technical details panel
- Professional footer

---

# 📊 Prediction Dashboard

After uploading an image, the application displays:

### 🏆 Best Prediction

Displays the class having the highest confidence score.

---

### 📈 Confidence Score

Shows the confidence percentage predicted by the model.

---

### ⚡ Inference Time

Measures the amount of time required by MobileNetV2 to classify the uploaded image.

---

### 📋 Top 5 Predictions

Displays the five most probable ImageNet classes together with confidence scores.

---

### 📥 Download Report

The prediction results can be exported as a CSV report directly from the application.

---

### 🕘 Session History

The application stores all predictions made during the current session for quick reference.

---

# 📂 Project Structure

```text
image-recognition/
│
├── Notebook.ipynb
├── app.py
├── requirements.txt
├── sample_image.jpg
├── README.md
└── screenshots/
```

---

# 📄 File Description

| File | Description |
|------|-------------|
| Notebook.ipynb | Complete development notebook including preprocessing, model implementation and Streamlit preparation |
| app.py | Streamlit web application with modern UI |
| requirements.txt | Python dependencies |
| sample_image.jpg | Sample image for testing |
| README.md | Project documentation |

---

# 📦 Requirements

```text
streamlit
tensorflow
numpy
Pillow
pandas
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Clone the repository.

```bash
git clone <repository-url>
```

Move into the project directory.

```bash
cd image-recognition
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Run Streamlit.

```bash
streamlit run app.py
```

---

# 💻 How to Use

### Step 1

Launch the Streamlit application.

---

### Step 2

Upload an image.

Supported formats:

- JPG
- JPEG
- PNG

---

### Step 3

The application automatically:

- Converts the image to RGB
- Resizes it to 224 × 224
- Applies MobileNetV2 preprocessing
- Performs image classification

---

### Step 4

The prediction dashboard displays:

- Best prediction
- Confidence
- Inference time
- Top five predictions

---

### Step 5

Download the prediction report or continue testing more images.

---

# ⚡ Performance

Model

- MobileNetV2

Input Size

- 224 × 224

Output Classes

- 1000

Prediction Type

- Image Classification

Framework

- TensorFlow

Deployment

- Streamlit

---

# 📚 What I Learned

Through this project, I gained practical experience with:

- Deep Learning fundamentals
- Convolutional Neural Networks
- Transfer Learning
- MobileNetV2
- TensorFlow/Keras
- Image preprocessing
- Image classification
- Streamlit application development
- UI/UX design for AI applications
- Session state management
- CSV report generation
- Model deployment
- Git & GitHub workflow

---

# 🚧 Future Improvements

I plan to further enhance this project by adding:

- Object Detection
- Image Segmentation
- Drag-and-drop upload
- Camera capture support
- Batch image prediction
- Dark/Light theme switch
- Prediction analytics dashboard
- Custom model training
- Support for additional pretrained models
- Docker deployment

---

# 👨‍💻 Author

**Shivam Pandey**

B.Tech — Computer Science & Engineering  
Specialization: Artificial Intelligence & Machine Learning

---

# 📜 Project Ownership

I independently developed this project as part of my Artificial Intelligence and Machine Learning portfolio.

The project demonstrates my understanding of:

- Deep Learning
- Transfer Learning
- Image Classification
- TensorFlow
- Streamlit
- Python
- AI Application Development
- GitHub Project Management

From preprocessing images to building a modern interactive web interface, every stage of this project reflects my practical implementation of an end-to-end AI application.

---

## 👨‍💻 Author

**Shivam Pandey**

B.Tech – Computer Science & Engineering (AI & ML)
