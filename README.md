# Sign Language Recognition

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 Overview

Sign Language Recognition is an advanced deep learning application designed to recognize and interpret American Sign Language (ASL) gestures in real-time. This project leverages state-of-the-art computer vision and neural network architectures to bridge communication gaps and promote accessibility for the deaf and hard-of-hearing community.

The system can identify 29 different hand gestures representing the complete ASL alphabet, including special characters like `space`, `nothing`, and `del`. By combining the power of transfer learning with the Xception architecture, this project achieves high accuracy while maintaining computational efficiency.

## ✨ Features

- **Real-Time Gesture Recognition**: Capture and classify ASL gestures using your webcam with OpenCV integration
- **Web-Based Interface**: User-friendly Streamlit application for uploading and analyzing static images
- **High Accuracy Model**: Utilizing the Xception architecture with transfer learning for superior performance
- **Confidence Scoring**: Detailed probability distribution across all 29 gesture classes
- **Interactive Visualization**: Real-time feedback with bounding boxes and prediction overlays
- **Lightweight Deployment**: Optimized for both CPU and GPU inference
- **Easy Integration**: Modular design allows easy integration into larger applications

## 🎬 Demo

### Real-Time Recognition
The application processes webcam feed in real-time, detecting hand gestures and displaying predictions with confidence scores instantly.

### Image Upload Interface
Users can upload static images through the Streamlit web interface to receive gesture predictions along with probability distributions for all classes.

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **TensorFlow 2.x / Keras**: Deep learning framework for model training and inference
- **Xception Architecture**: Pre-trained CNN model serving as the backbone for transfer learning

### Computer Vision & Processing
- **OpenCV**: Real-time video capture and image preprocessing
- **NumPy**: Numerical computing and array operations
- **PIL (Pillow)**: Image manipulation and format conversions

### Web Interface
- **Streamlit**: Interactive web application framework for model deployment
- **Pandas**: Data manipulation and probability table display

## 📊 Dataset

**Source**: [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

### Dataset Characteristics
- **Total Classes**: 29 (A-Z, space, delete, nothing)
- **Image Format**: RGB images (200x200 pixels)
- **Training Samples**: ~87,000 images
- **Diversity**: Multiple skin tones, backgrounds, and lighting conditions
- **Organization**: Structured by class folders for easy loading

### Data Preprocessing
- Image resizing to model input dimensions (299x299 for Xception)
- Normalization of pixel values to [0, 1] range
- Data augmentation techniques including rotation, zoom, and flip
- Train-validation-test split for robust model evaluation

## 📁 Project Structure

```
sign-language-recognition/
│
├── asl_recognition.py          # Real-time webcam recognition script
├── streamlit_app.py            # Streamlit web application
├── train_model.py              # Model training pipeline
├── model/
│   ├── asl_model.h5           # Trained Keras model
│   └── class_labels.json      # Class name mappings
│
├── data/
│   ├── train/                 # Training dataset
│   ├── validation/            # Validation dataset
│   └── test/                  # Test dataset
│
├── utils/
│   ├── preprocessing.py       # Image preprocessing utilities
│   └── model_utils.py         # Model loading and prediction helpers
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Data exploration and visualization
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── LICENSE                    # MIT License
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Webcam (for real-time recognition)
- 4GB+ RAM recommended

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/arpanpramanik2003/sign-language-recognition.git
   cd sign-language-recognition
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Model** (if not included)
   - Place the `asl_model.h5` file in the `model/` directory
   - Alternatively, train your own model using `train_model.py`

## 💻 Usage

### Option 1: Real-Time Webcam Recognition

Run the real-time gesture recognition script:

```bash
python asl_recognition.py
```

**Controls:**
- Press `q` to quit
- Press `s` to save the current frame
- Ensure good lighting for optimal performance

### Option 2: Streamlit Web Application

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`

**Features:**
- Upload image files (JPG, PNG, JPEG)
- View prediction results with confidence scores
- Explore probability distributions across all classes
- Download prediction reports

### Option 3: Training Your Own Model

To train the model from scratch:

```bash
python train_model.py --epochs 50 --batch_size 32
```

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Initial learning rate (default: 0.0001)
- `--data_path`: Path to dataset directory

## 📋 Requirements

```txt
tensorflow>=2.8.0
keras>=2.8.0
opencv-python>=4.5.0
streamlit>=1.10.0
numpy>=1.21.0
pillow>=9.0.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
```

### Optional Dependencies
- **CUDA Toolkit** (for GPU acceleration)
- **cuDNN** (for TensorFlow GPU support)

## 🤝 Contributing

Contributions are welcome and greatly appreciated! Here's how you can contribute:

### Ways to Contribute
1. **Report Bugs**: Open an issue describing the bug and how to reproduce it
2. **Suggest Features**: Propose new features or enhancements
3. **Submit Pull Requests**: Fix bugs, add features, or improve documentation
4. **Improve Documentation**: Help make the documentation clearer and more comprehensive
5. **Share Dataset**: Contribute additional training data for model improvement

### Contribution Guidelines

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sign-language-recognition.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Make Your Changes**
   - Write clean, documented code
   - Follow PEP 8 style guidelines
   - Add tests for new features

4. **Commit Your Changes**
   ```bash
   git commit -m "Add some AmazingFeature"
   ```

5. **Push to Your Fork**
   ```bash
   git push origin feature/AmazingFeature
   ```

6. **Open a Pull Request**
   - Provide a clear description of the changes
   - Reference any related issues

### Code of Conduct
Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive environment for all contributors.

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ❌ Liability disclaimer
- ❌ Warranty disclaimer

## 📧 Contact

**Project Maintainer**: Arpan Pramanik

- **GitHub**: [@arpanpramanik2003](https://github.com/arpanpramanik2003)
- **Repository**: [sign-language-recognition](https://github.com/arpanpramanik2003/sign-language-recognition)
- **Issues**: [Report a bug or request a feature](https://github.com/arpanpramanik2003/sign-language-recognition/issues)

### Connect & Collaborate
Feel free to reach out for:
- Technical questions or support
- Collaboration opportunities
- Feature requests or suggestions
- General feedback

---

## 🙏 Acknowledgments

- **Dataset**: Thanks to the Kaggle community and the ASL Alphabet dataset contributors
- **TensorFlow Team**: For the excellent deep learning framework
- **Xception Architecture**: François Chollet for the Xception model
- **OpenCV Community**: For robust computer vision tools
- **Open Source Community**: For inspiration and support

---

## 🌟 Star This Repository

If you find this project useful, please consider giving it a ⭐ on GitHub. It helps others discover the project and motivates continued development!

---

**Made with ❤️ for accessibility and innovation**
