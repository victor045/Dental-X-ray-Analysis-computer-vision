# 🦷 Dental X-ray Analysis App

A Streamlit web application for analyzing dental X-ray images using deep learning models for both classification and detection.

## ✨ Features

- **🖼️ Image Upload**: Upload dental X-ray images (PNG, JPG, JPEG)
- **🔍 Classification**: Classify dental conditions (Healthy, Cavity, Crown, Filling)
- **📍 Detection**: Detect individual teeth with bounding boxes
- **🌐 Bilingual**: English and Spanish interface
- **🎨 Color-coded Results**: Different colors for different classifications
- **📊 Confidence Scores**: View prediction confidence for each result

## 🚀 Quick Start

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**:
   ```bash
   streamlit run correct_streamlit_app.py
   ```

3. **Open in browser**: http://localhost:8501

### Deploy to Streamlit Cloud

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add dental X-ray analysis app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select this repository
   - Set main file to: `correct_streamlit_app.py`
   - Click "Deploy"

## 📁 Project Structure

```
dentexmodel/
├── correct_streamlit_app.py    # Main Streamlit application
├── requirements.txt            # Python dependencies
├── data/                       # Data and model files
│   └── model/
│       ├── FancyLR/           # Classification model
│       └── Toothdetector/     # Detection model
├── src/                        # Source code
│   └── dentexmodel/
└── README.md                   # This file
```

## 🧠 Models Used

- **Classification**: PyTorch Lightning ResNet50 model
- **Detection**: Detectron2 Faster R-CNN model
- **Preprocessing**: ImageNet normalization

## 🌐 Languages

- **English**: Default interface
- **Spanish**: Select "Español" in the sidebar

## 🎨 Color Coding

- **🟢 Green**: Healthy/Sano
- **🔴 Red**: Cavity/Caries
- **🔵 Blue**: Crown/Corona
- **🟠 Orange**: Filling/Empaste

## 📝 License

This project is part of the DentexModel research project.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!
