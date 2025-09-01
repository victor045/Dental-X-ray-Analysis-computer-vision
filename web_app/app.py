import os
import io
import base64
import logging
from datetime import datetime
from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import cv2

# Import your models
import sys
sys.path.append('../src')
from dentexmodel.models.toothmodel_basic import ToothModel, ResNet50Model
from dentexmodel.models.toothmodel_fancy import ToothModel as FancyToothModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Class labels (you may need to adjust these based on your training data)
CLASS_LABELS = {
    0: "Healthy",
    1: "Cavity", 
    2: "Crown",
    3: "Filling"
}

# Global model variables
classification_model = None
detection_model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load the trained models"""
    global classification_model, detection_model
    
    try:
        # Load classification model
        model_path = '../data/model/classification_model.pth'
        if os.path.exists(model_path):
            classification_model = ResNet50Model(n_outputs=len(CLASS_LABELS)).create_model()
            classification_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            classification_model.eval()
            logger.info("Classification model loaded successfully")
        
        # Load detection model (if available)
        detection_path = '../data/model/detection_model.pth'
        if os.path.exists(detection_path):
            # You'll need to implement detection model loading based on your detection model
            logger.info("Detection model loaded successfully")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model inference"""
    # Resize to standard size
    image = image.resize((224, 224))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(np.array(image)).float()
    image_tensor = image_tensor.permute(2, 0, 1) / 255.0
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def classify_image(image: Image.Image) -> Dict[str, Any]:
    """Perform classification on the image"""
    if classification_model is None:
        return {"error": "Classification model not loaded"}
    
    try:
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Perform inference
        with torch.no_grad():
            outputs = classification_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        
        results = {
            "predicted_class": CLASS_LABELS[predicted_class],
            "confidence": round(confidence, 4),
            "all_predictions": [
                {
                    "class": CLASS_LABELS[idx.item()],
                    "confidence": round(prob.item(), 4)
                }
                for prob, idx in zip(top3_probs, top3_indices)
            ]
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        return {"error": str(e)}

def detect_teeth(image: Image.Image) -> Dict[str, Any]:
    """Perform tooth detection on the image"""
    # This is a placeholder - you'll need to implement based on your detection model
    # For now, we'll return a simple bounding box detection
    
    try:
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Simple edge detection for demonstration
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 1000
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    "bbox": [x, y, x + w, y + h],
                    "confidence": 0.8,  # Placeholder confidence
                    "class": "tooth"
                })
        
        return {
            "detections": detections,
            "num_detections": len(detections)
        }
        
    except Exception as e:
        logger.error(f"Error in detection: {e}")
        return {"error": str(e)}

def create_annotated_image(image: Image.Image, classification_result: Dict, detection_result: Dict) -> Image.Image:
    """Create an annotated image with classification and detection results"""
    # Create a copy of the image for annotation
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Add classification results
    if "predicted_class" in classification_result:
        text = f"Classification: {classification_result['predicted_class']} ({classification_result['confidence']:.2%})"
        draw.text((10, 10), text, fill="red", font=font)
    
    # Add detection results
    if "detections" in detection_result:
        for i, detection in enumerate(detection_result["detections"]):
            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            
            # Add label
            label = f"Tooth {i+1}"
            draw.text((x1, y1-25), label, fill="green", font=font)
    
    return annotated_image

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Read and process image
        image = Image.open(file.stream)
        
        # Perform analysis
        classification_result = classify_image(image)
        detection_result = detect_teeth(image)
        
        # Create annotated image
        annotated_image = create_annotated_image(image, classification_result, detection_result)
        
        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"result_{timestamp}.png"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        annotated_image.save(result_path)
        
        # Convert annotated image to base64 for display
        buffer = io.BytesIO()
        annotated_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Prepare response
        response = {
            'success': True,
            'classification': classification_result,
            'detection': detection_result,
            'annotated_image': img_str,
            'result_filename': result_filename
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_result(filename):
    """Download result image"""
    try:
        return send_file(
            os.path.join(app.config['RESULTS_FOLDER'], filename),
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'classification_model_loaded': classification_model is not None,
        'detection_model_loaded': detection_model is not None
    })

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)

