# 🦷 Dental X-ray Analysis - Streamlit Cloud Deployment

This is a Streamlit Cloud optimized version of the dental X-ray analysis app that uses **Ultralytics YOLO** for detection and **Lightning** for classification, running entirely on CPU for reliable cloud deployment.

## 🚀 Key Features

- **CPU-Only**: Optimized for Streamlit Cloud with no GPU dependencies
- **YOLO Detection**: Uses Ultralytics YOLO instead of Detectron2 for better cloud compatibility
- **Lightning Classification**: Keeps the original Lightning classification model
- **Bilingual**: English and Spanish support
- **Performance Monitoring**: Real-time CPU and memory usage tracking
- **Auto-Download**: Automatically downloads models from S3

## 📋 Requirements

The app uses the following optimized dependencies:

```txt
streamlit>=1.36
pillow
numpy
pandas
albumentations
psutil
opencv-python-headless  # Headless version for cloud
requests
torch==2.2.1+cpu        # CPU-only PyTorch
torchvision==0.17.1+cpu # CPU-only TorchVision
ultralytics==8.2.0      # YOLO detection
lightning               # Classification model
torchmetrics
```

## 🌐 Streamlit Cloud Deployment

### 1. **Prepare Your Repository**

Ensure your repository has the following structure:
```
your-repo/
├── streamlit_cloud_app.py    # Main app file
├── requirements.txt          # Dependencies
├── src/                      # Your dentexmodel package
│   └── dentexmodel/
├── data/                     # Model directory (will be created)
└── README.md
```

### 2. **Model Files**

The app will automatically download models from S3:
- **Classification**: `toothmodel_fancy_40.ckpt` (Lightning checkpoint)
- **Detection**: `tooth_yolo.pt` (YOLO weights)

Update the URLs in `streamlit_cloud_app.py`:
```python
CLASS_CKPT_URL = "https://your-s3-bucket.s3.amazonaws.com/toothmodel_fancy_40.ckpt"
YOLO_WEIGHTS_URL = "https://your-s3-bucket.s3.amazonaws.com/tooth_yolo.pt"
```

### 3. **Deploy to Streamlit Cloud**

1. **Push to GitHub**: Commit and push your code to GitHub
2. **Connect to Streamlit Cloud**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set the main file path to: `streamlit_cloud_app.py`
3. **Deploy**: Click "Deploy App"

## 🔧 Local Development

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_cloud_app.py
```

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Model Architecture

### Classification Model (Lightning)
- **Framework**: PyTorch Lightning
- **Architecture**: ResNet-50 with custom head
- **Classes**: 4 (Healthy, Cavity, Crown, Filling)
- **Input**: 224x224 RGB images
- **Output**: Classification probabilities

### Detection Model (YOLO)
- **Framework**: Ultralytics YOLO
- **Architecture**: YOLOv8 (or your custom model)
- **Classes**: 1 (tooth)
- **Input**: Variable size RGB images
- **Output**: Bounding boxes with confidence scores

## 🎯 Usage

1. **Upload Image**: Select a dental X-ray image (PNG, JPG, JPEG)
2. **Choose Analysis**: 
   - Detection Only: Find teeth locations
   - Classification Only: Classify the entire image
   - Both: Detect teeth and classify each tooth
3. **Adjust Settings**: Set confidence threshold for detection
4. **View Results**: See predictions, confidence scores, and annotated image

## 🔍 Performance Monitoring

The app includes real-time monitoring:
- **CPU Usage**: Current CPU utilization
- **Memory Usage**: RAM usage and availability
- **Inference Time**: Time taken for each prediction
- **Device Verification**: Confirms CPU usage

## 🛠️ Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check S3 URLs are accessible
   - Verify network connectivity
   - Check file permissions

2. **Memory Issues**
   - Reduce image size
   - Lower batch size
   - Monitor memory usage in sidebar

3. **Slow Performance**
   - Normal for CPU-only deployment
   - Consider image resizing
   - Check system resources

### Debug Mode

Add debug information by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Optimization

### For Streamlit Cloud
- **CPU-Only**: No GPU dependencies for reliability
- **Headless OpenCV**: Lighter installation
- **Model Caching**: Uses `@st.cache_resource`
- **Efficient Processing**: Optimized image preprocessing

### For Local Development
- **GPU Support**: Can be enabled by modifying device settings
- **Batch Processing**: Can be optimized for multiple images
- **Model Optimization**: Consider quantization for faster inference

## 🔄 Migration from Detectron2

This app replaces Detectron2 with Ultralytics YOLO:

### Before (Detectron2)
```python
from detectron2.engine import DefaultPredictor
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### After (YOLO)
```python
from ultralytics import YOLO
yolo_model = YOLO(weights_path)
yolo_model.to('cpu')
```

## 📝 License

This project follows the same license as the original dentexmodel project.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally and on Streamlit Cloud
5. Submit a pull request

## 📞 Support

For issues related to:
- **Streamlit Cloud**: Check [Streamlit documentation](https://docs.streamlit.io)
- **YOLO**: Check [Ultralytics documentation](https://docs.ultralytics.com)
- **Lightning**: Check [PyTorch Lightning documentation](https://lightning.ai/docs)

---

**Note**: This app is optimized for Streamlit Cloud deployment and may not utilize GPU acceleration. For local development with GPU support, use the original `correct_streamlit_app.py`.


