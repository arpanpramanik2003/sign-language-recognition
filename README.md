[🔗 Dataset Link (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

---

## 📌 Project Overview  
This project is a **deep learning-based ASL (American Sign Language) gesture recognition system** that identifies 29 different hand gestures representing the ASL alphabet, including `space`, `nothing`, and `del`. The system includes:

- A **real-time interface** using OpenCV and webcam to detect hand gestures.
- A **Streamlit web app** to upload an image and predict the ASL gesture class with confidence.

The model is built using **TensorFlow** and the **Xception** architecture for high accuracy and generalization.

---

## 🖥️ Tech Stack  
- **Python** (Core programming language)  
- **TensorFlow/Keras** (For model training and inference)  
- **Xception** (Pre-trained CNN model for transfer learning)  
- **OpenCV** (For real-time webcam capture)  
- **Streamlit** (For building the web interface)  
- **NumPy** (For array manipulations)  
- **PIL** (For image preprocessing)

---

## 🎯 Features  
- Predict ASL gestures in real-time using a webcam  
- Upload and classify static hand gesture images  
- Confidence score for prediction  
- Table view of all class probabilities (in Streamlit)  
- Lightweight and fast model using transfer learning  

---

## 🚀 Installation & Usage  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/arpanpramanik2003/sign language recognition.git  
cd sign language recognition
