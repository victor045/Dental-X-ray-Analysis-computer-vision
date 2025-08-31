# Dental Image Analysis Web Application

A cloud-based web application for AI-powered dental image analysis, providing both classification and detection capabilities.

## Features

- **Image Upload**: Drag-and-drop or click-to-upload interface
- **Classification**: Identify dental conditions (Healthy, Cavity, Crown, Filling)
- **Detection**: Detect and locate teeth in images
- **Annotated Results**: View results with bounding boxes and labels
- **Download Results**: Save annotated images
- **Modern UI**: Responsive design with beautiful gradients

## Quick Start

### Option 1: Run with Docker (Recommended)

1. **Build and run the application:**
   ```bash
   cd web_app
   docker-compose up --build
   ```

2. **Access the application:**
   - Open your browser and go to: `http://localhost:5000`

### Option 2: Run Locally

1. **Install dependencies:**
   ```bash
   cd web_app
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Access the application:**
   - Open your browser and go to: `http://localhost:5000`

## Prerequisites

### For Local Development:
- Python 3.8+
- pip
- Trained models in `../data/model/` directory

### For Docker:
- Docker
- Docker Compose

## Model Setup

The application expects trained models in the following locations:

```
../data/model/
├── classification_model.pth    # Classification model weights
└── detection_model.pth         # Detection model weights (optional)
```

### Training Your Own Models

1. **Classification Model:**
   - Use the notebooks in `../notebooks/classification/`
   - Train using `06_model_training_basic.ipynb` or `07_model_training_fancy.ipynb`
   - Save the model as `classification_model.pth`

2. **Detection Model:**
   - Use the notebooks in `../notebooks/detect-segment/`
   - Implement detection model training
   - Save the model as `detection_model.pth`

## API Endpoints

### POST `/upload`
Upload and analyze an image.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "success": true,
  "classification": {
    "predicted_class": "Healthy",
    "confidence": 0.95,
    "all_predictions": [
      {"class": "Healthy", "confidence": 0.95},
      {"class": "Cavity", "confidence": 0.03},
      {"class": "Crown", "confidence": 0.02}
    ]
  },
  "detection": {
    "detections": [
      {
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.8,
        "class": "tooth"
      }
    ],
    "num_detections": 1
  },
  "annotated_image": "base64_encoded_image",
  "result_filename": "result_20230830_123456.png"
}
```

### GET `/download/<filename>`
Download a result image.

### GET `/health`
Health check endpoint.

## Deployment Options

### 1. Local Development
```bash
python app.py
```

### 2. Docker
```bash
docker-compose up --build
```

### 3. Cloud Deployment

#### Heroku
1. Create a `Procfile`:
   ```
   web: gunicorn app:app
   ```

2. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

#### AWS/GCP/Azure
1. Build Docker image
2. Push to container registry
3. Deploy to container service

### 4. Production Considerations

- **Environment Variables:**
  ```bash
  export FLASK_ENV=production
  export FLASK_DEBUG=0
  ```

- **Security:**
  - Use HTTPS
  - Implement authentication
  - Rate limiting
  - Input validation

- **Performance:**
  - Use multiple workers with Gunicorn
  - Implement caching
  - Use CDN for static files

## Customization

### Adding New Classes
1. Update `CLASS_LABELS` in `app.py`
2. Retrain your model with the new classes
3. Update the model file

### Modifying Detection
1. Implement your detection model in `detect_teeth()` function
2. Update the detection logic as needed

### UI Customization
1. Modify `templates/index.html`
2. Update CSS styles
3. Add new features to the interface

## Troubleshooting

### Common Issues

1. **Model not loading:**
   - Check if model files exist in `../data/model/`
   - Verify model file paths in `app.py`

2. **Import errors:**
   - Ensure all dependencies are installed
   - Check Python path includes `../src`

3. **Docker issues:**
   - Rebuild the image: `docker-compose build --no-cache`
   - Check container logs: `docker-compose logs`

4. **Memory issues:**
   - Reduce image size limits
   - Use smaller model variants
   - Implement image resizing

### Logs
Check application logs for detailed error information:
```bash
docker-compose logs web-app
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

