###🪙 **Project CPE431 - Coin Detection and Counting**

###🔍 **Project Overview**
This project is a system for detecting and counting coins using Computer Vision and Machine Learning. It processes coin images to identify their type and count the number of coins detected.

###🚀 **Key Features**
📸 Coin Detection from images
🔢 Coin Counting for different types
🎯 Coin Classification based on size and features
📊 Real-Time Data Display of detected coins

###🛠️ **Technologies Used**
Python 🐍
OpenCV 🎥 (for image processing)
TensorFlow/PyTorch 🧠 (for Machine Learning)
NumPy & Pandas 📊 (for data handling)

### 🛠️ Required Packages for **TensorFlow Object Detection API**  

To set up TensorFlow Object Detection API, ensure you have the following installed:  

---

### ✅ **1. Install Microsoft Visual C++ Redistributable (Latest)**  
🔗 Download: [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)  
📌 Required for running TensorFlow with GPU acceleration on Windows.  

---

### ✅ **2. Install CUDA and cuDNN for GPU Support**  
📌 TensorFlow 2.8 supports **CUDA 11.2** and **cuDNN 8.1**.  

#### **🔹 Install CUDA Toolkit**  
🔗 Download: [CUDA Toolkit 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive)  

#### **🔹 Install cuDNN (for CUDA 11.2)**  
```sh
conda install -c conda-forge cudnn=8.1
```

---

### ✅ **3. Install TensorFlow with GPU Support**  
```sh
pip install tensorflow-gpu==2.8.*
```

---

### ✅ **4. Install OpenCV for Image Processing**  
```sh
pip install opencv-python
pip install opencv-contrib-python
```

---

### ✅ **5. Install Flask for API Development**  
```sh
pip install Flask
pip install Flask-Navigation
```

---

### ✅ **6. Install Additional Dependencies**  
```sh
pip install shapely         # Geometry operations (for bounding boxes)
pip install protobuf==3.20.*  # Required for TensorFlow
pip install pillow         # Image processing
pip install lxml           # XML parsing (for label maps)
pip install Cython         # Performance optimization
```

---

### ✅ **7. Verify Installation**  
After installation, verify that TensorFlow detects your GPU:  
```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

🔹 If this prints `"Num GPUs Available: 1"` (or more), TensorFlow is using your GPU correctly. 🚀  


### 🎯 **Done!**  
You’re now ready to use **TensorFlow Object Detection API** with GPU acceleration! 


### 🛠️ **How to Use Object Detection with Flask**  

This guide explains how to set up and use **Object Detection with Flask**.  

---

## **1️⃣ Install Required Packages**  
Run the following commands to install the necessary dependencies:  
```sh
pip install flask
pip install flask-cors
pip install tensorflow
pip install opencv-python
pip install pillow
pip install numpy
pip install protobuf==3.20.*
```

---

## **2️⃣ Project Structure**  

```
object_detection_flask/
│── models/                  # Pre-trained model files  
│── static/                  # Folder for input images  
│── templates/               # HTML templates (if using UI)  
│── app.py                   # Flask server  
│── detect.py                # Object detection script  
│── count_object.py          # Count object script (standalone)
```

---

## **5️⃣ Run the Flask Server**  
```sh
python app.py
```
The server will start at `http://127.0.0.1:5000/`.

---


## 🎯 **Done!**  
Now, your Flask API can process images and detect objects using TensorFlow Object Detection API! 🚀🔥
