# smart-waste-management
Smart Waste Classification System (DIP Project)

## 📌 Overview
The **Smart Waste Management System** is an **image-processing-based project** that classifies waste into categories such as **plastic, glass, paper, and metal**.  
By automating waste classification, this project supports **sustainable recycling and efficient waste management**.

---

## ⚙️ Features
- 🖼️ **Image Preprocessing**: resizing, grayscale conversion, Gaussian blur, thresholding.  
- ✨ **Contour Detection**: isolates the waste item by detecting its outline.  
- 🤖 **Deep Learning Classification**: identifies the type of waste using a trained CNN model.  
- 📊 **Visualization**: displays processed images and predicted labels.  

---

## 🚀 Tech Stack
- **Python 3.x**
- **OpenCV** – Image preprocessing & contour detection  
- **TensorFlow/Keras** – Deep learning model  
- **NumPy & Pandas** – Data handling  
- **Matplotlib** – Visualization  

---

## 📂 Project Structure
📁 SmartWasteManagement
┣ 📂 data/ # Dataset (images of different waste categories)
┣ 📂 models/ # Saved trained model (.h5/.tflite)
┣ 📂 notebooks/ # Jupyter notebooks for training & testing
┣ 📂 src/ # Source code
┣ README.md
┣ requirements.txt
┗ LICENSE
