# 🦷 Dental X-ray Analysis - Hugging Face Spaces Deployment

This repository is optimized for deployment on Hugging Face Spaces using Docker and Streamlit.

## 🚀 Quick Deployment

### 1. Create a Hugging Face Space
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - **Owner**: Your username
   - **Space name**: `dental-xray-analysis`
   - **License**: MIT
   - **SDK**: **Docker**
   - **Hardware**: **CPU basic** (free tier)

### 2. Repository Structure
```
repo/
├─ Dockerfile                 # Docker configuration
├─ requirements.txt           # Python dependencies (CPU-optimized)
├─ .dockerignore             # Files to exclude from Docker build
├─ streamlit_cloud_app.py    # Main Streamlit application
├─ src/
│  └─ dentexmodel/
│     ├─ __init__.py
│     ├─ models/
│     │   └─ toothmodel_fancy.py
│     ├─ imageproc.py
│     ├─ torchdataset.py
│     └─ fileutils.py
└─ (models auto-downloaded at runtime)
```

### 3. Environment Variables (Optional)
If you have private model URLs, add them as Repository secrets:
- `YOLO_WEIGHTS_URL`: URL to your trained YOLO model
- `CLASS_CKPT_URL`: URL to your classification checkpoint

## 🔧 Technical Details

### CPU Optimization
- **PyTorch**: CPU-only wheels (`torch==2.2.2+cpu`)
- **OpenCV**: Headless version (`opencv-python-headless`)
- **Memory**: Optimized for 16GB RAM limit
- **Port**: Uses `$PORT` environment variable (HF Spaces requirement)

### Model Loading
- **YOLO Model**: Auto-downloads trained `tooth_yolo.pt` (99.3% mAP50)
- **Classification**: Lightning model with CPU compatibility
- **Fallback**: Graceful handling if models unavailable

### Performance
- **Detection**: YOLOv8 trained on dental dataset
- **Classification**: Lightning-based tooth condition classifier
- **Inference**: Optimized for CPU-only environments

## 📊 Features

### Analysis Types
1. **Detection Only**: Find and locate teeth in X-rays
2. **Classification Only**: Classify tooth conditions (Healthy, Cavity, Crown, Filling)
3. **Both**: Complete analysis with per-tooth classification

### Languages
- **English**: Primary interface
- **Spanish**: Full translation support

### Output
- **Bounding Boxes**: Precise tooth localization
- **Confidence Scores**: Detection and classification confidence
- **Annotated Images**: Visual results with predictions
- **System Monitoring**: CPU and memory usage

## 🛠️ Local Development

### Build Docker Image
```bash
docker build -t dental-xray-analysis .
```

### Run Locally
```bash
docker run -p 8501:8501 dental-xray-analysis
```

### Test with Sample Images
1. Upload dental X-ray images
2. Adjust confidence threshold (0.1-0.9)
3. View real-time analysis results

## 🔍 Troubleshooting

### Common Issues
1. **Memory OOM**: Reduce image size in YOLO config
2. **Model Download Failures**: Check URL accessibility
3. **Port Issues**: Ensure using `$PORT` environment variable

### Logs
- Check Streamlit logs in HF Spaces interface
- Monitor resource usage in Space settings

## 📈 Performance Metrics

### YOLO Model Performance
- **mAP50**: 99.3%
- **mAP50-95**: 60.9%
- **Precision**: 98.6%
- **Recall**: 97.2%

### Hardware Requirements
- **CPU**: 2 vCPU (HF Spaces free tier)
- **RAM**: 16GB
- **Storage**: Models auto-download (~100MB)

## 🎯 Deployment Checklist

- [ ] Repository structure matches template
- [ ] Dockerfile uses correct base image
- [ ] requirements.txt has CPU-only dependencies
- [ ] Models accessible via URLs
- [ ] Environment variables set (if needed)
- [ ] Port configuration uses `$PORT`
- [ ] .dockerignore excludes large files

## 📞 Support

For issues with:
- **HF Spaces**: Check [Spaces documentation](https://huggingface.co/docs/hub/spaces)
- **Docker**: Review Docker logs in Space settings
- **Model Performance**: Verify model URLs and accessibility

---

**Ready for deployment!** 🚀
