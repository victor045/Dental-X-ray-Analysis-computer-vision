#!/usr/bin/env python3
"""
Correct Streamlit app using the project's models exactly as in the notebooks
"""

import os
import sys
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import pandas as pd
import logging
import glob
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Set DATA_ROOT environment variable
os.environ['DATA_ROOT'] = os.path.join(os.getcwd(), 'data')

# Import project modules
import dentexmodel as dm
from dentexmodel.models.toothmodel_fancy import ToothModel
from dentexmodel.imageproc import ImageData
from dentexmodel.torchdataset import DatasetFromDF, load_and_process_image
from dentexmodel.fileutils import FileOP

# Albumentations for transforms
import albumentations as alb

# PyTorch imports
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language translations
TRANSLATIONS = {
    'en': {
        'title': '🦷 Dental X-ray Analysis',
        'subtitle': 'Upload a dental X-ray image for classification and detection analysis using the project\'s actual models.',
        'language_label': 'Language / Idioma:',
        'analysis_type_label': 'Select Analysis Type:',
        'detection_only': 'Detection Only',
        'classification_only': 'Classification Only',
        'both': 'Both Classification & Detection',
        'confidence_threshold': 'Detection Confidence Threshold:',
        'file_uploader': 'Choose a dental X-ray image...',
        'original_image': 'Original Image',
        'uploaded_xray': 'Uploaded X-ray',
        'analysis_results': 'Analysis Results',
        'classification': 'Classification',
        'confidence': 'Confidence',
        'all_predictions': 'All Predictions',
        'detection': 'Detection',
        'teeth_found': 'teeth found',
        'model_type': 'Model type',
        'top_detections': 'Top Detections',
        'per_tooth_completed': 'Per-tooth classification completed',
        'annotated_results': 'Annotated Results',
        'annotated_xray': 'Annotated X-ray with predictions',
        'loading_models': 'Loading models...',
        'performing_classification': 'Performing classification...',
        'performing_detection': 'Performing detection...',
        'classifying_teeth': 'Classifying individual teeth...',
        'models_not_loaded': 'Failed to load required models. Please ensure both classification and detection models are properly trained and available.',
        'classification_error': 'Classification error',
        'detection_error': 'Detection error',
        'tooth': 'Tooth',
        'bbox': 'bbox',
        'conf': 'conf',
        'healthy': 'Healthy',
        'cavity': 'Cavity',
        'crown': 'Crown',
        'filling': 'Filling'
    },
    'es': {
        'title': '🦷 Análisis de Radiografía Dental',
        'subtitle': 'Sube una imagen de radiografía dental para análisis de clasificación y detección usando los modelos reales del proyecto.',
        'language_label': 'Language / Idioma:',
        'analysis_type_label': 'Seleccionar Tipo de Análisis:',
        'detection_only': 'Solo Detección',
        'classification_only': 'Solo Clasificación',
        'both': 'Clasificación y Detección',
        'confidence_threshold': 'Umbral de Confianza de Detección:',
        'file_uploader': 'Elige una imagen de radiografía dental...',
        'original_image': 'Imagen Original',
        'uploaded_xray': 'Radiografía Subida',
        'analysis_results': 'Resultados del Análisis',
        'classification': 'Clasificación',
        'confidence': 'Confianza',
        'all_predictions': 'Todas las Predicciones',
        'detection': 'Detección',
        'teeth_found': 'dientes encontrados',
        'model_type': 'Tipo de modelo',
        'top_detections': 'Mejores Detecciones',
        'per_tooth_completed': 'Clasificación por diente completada',
        'annotated_results': 'Resultados Anotados',
        'annotated_xray': 'Radiografía anotada con predicciones',
        'loading_models': 'Cargando modelos...',
        'performing_classification': 'Realizando clasificación...',
        'performing_detection': 'Realizando detección...',
        'classifying_teeth': 'Clasificando dientes individuales...',
        'models_not_loaded': 'Error al cargar los modelos requeridos. Asegúrate de que tanto el modelo de clasificación como el de detección estén correctamente entrenados y disponibles.',
        'classification_error': 'Error de clasificación',
        'detection_error': 'Error de detección',
        'tooth': 'Diente',
        'bbox': 'caja',
        'conf': 'conf',
        'healthy': 'Sano',
        'cavity': 'Caries',
        'crown': 'Corona',
        'filling': 'Empaste'
    }
}

# Class labels from the project
CLASS_LABELS = {
    0: "Healthy",
    1: "Cavity", 
    2: "Crown",
    3: "Filling"
}

# Spanish class labels
CLASS_LABELS_ES = {
    0: "Sano",
    1: "Caries", 
    2: "Corona",
    3: "Empaste"
}

# Colors for different classifications
CLASSIFICATION_COLORS = {
    "Healthy": "green",
    "Sano": "green",
    "Cavity": "red", 
    "Caries": "red",
    "Crown": "blue",
    "Corona": "blue",
    "Filling": "orange",
    "Empaste": "orange"
}

# Model URLs for auto-download
CLASS_CKPT_URL = "https://dsets.s3.amazonaws.com/dentex/toothmodel_fancy_40.ckpt"
DET_WEIGHTS_URL = "https://dsets.s3.amazonaws.com/dentex/toothdetection_3K.pth"

def _download(url: str, dst: str):
    """Download file from URL to destination"""
    import requests
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return dst

@st.cache_resource
def load_models():
    """Load classification (Lightning) and detection (Detectron2) models - ONLY proper models."""
    models = {}
    data_root = os.environ.get("DATA_ROOT", "./data")
    model_dir = os.path.join(data_root, "model")

    # ---------- Classification (Lightning ToothModel) ----------
    cls_ckpt = os.path.join(model_dir, "FancyLR", "version_1", "checkpoints", "last.ckpt")
    if not os.path.exists(cls_ckpt):
        # Use the same public checkpoint your notebooks reference
        try:
            st.info("Downloading classification checkpoint…")
            _download(CLASS_CKPT_URL, cls_ckpt)
        except Exception as e:
            st.error(f"Could not fetch classification ckpt: {e}")

    if os.path.exists(cls_ckpt):
        try:
            # Load the original Lightning module
            from dentexmodel.models.toothmodel_fancy import ToothModel as LToothModel
            from dentexmodel.torchdataset import DatasetFromDF
            import albumentations as alb
            from dentexmodel.imageproc import ImageData
            import pandas as pd

            # Create a proper dummy dataset that doesn't require actual files
            class DummyDataset:
                def __init__(self):
                    self.data = []
                
                def __len__(self):
                    return 1
                
                def __getitem__(self, idx):
                    # Return dummy data
                    dummy_image = torch.randn(3, 224, 224)
                    dummy_label = torch.tensor(0)
                    return dummy_image, dummy_label

            # Create the model with dummy datasets
            classification_model = LToothModel(
                train_dataset=DummyDataset(),
                val_dataset=DummyDataset(),
                test_dataset=DummyDataset(),
                batch_size=1,
                num_classes=4,
                num_workers=0,
                lr=1e-3
            )
            
            # Load the checkpoint
            checkpoint = torch.load(cls_ckpt, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            # Fix the state dict keys - add "model." prefix to match ToothModel structure
            fixed_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('model.'):
                    fixed_key = f'model.{key}'
                else:
                    fixed_key = key
                fixed_state_dict[fixed_key] = value
            
            # Load the fixed state dict
            classification_model.load_state_dict(fixed_state_dict, strict=False)
            classification_model.eval()
            models["classification"] = classification_model
            models["class_model_type"] = "lightning"
            st.success("✅ Classification model ready (Lightning)")

        except Exception as e:
            st.error(f"Classification load failed: {e}")
    else:
        st.error("❌ Classification checkpoint not found. Please train the classification model first.")

    # ---------- Detection (Detectron2 Faster R-CNN) ----------
    det_pth = os.path.join(model_dir, "Toothdetector", "version_1", "model_final.pth")
    if not os.path.exists(det_pth):
        try:
            st.info("Downloading detection weights…")
            _download(DET_WEIGHTS_URL, det_pth)
        except Exception as e:
            st.error(f"Could not fetch detection weights: {e}")

    try:
        if os.path.exists(det_pth):
            from detectron2.config import get_cfg
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor

            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            cfg.MODEL.WEIGHTS = det_pth
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.30
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

            models["detector"] = DefaultPredictor(cfg)
            models["detector_type"] = "detectron2"
            st.success(f"✅ Detection model ready ({cfg.MODEL.DEVICE.upper()})")
        else:
            st.error("❌ Detection model not found. Please train the detection model first.")
    except Exception as e:
        st.error(f"Detectron2 not available or failed to initialize: {e}")

    return models

def preprocess_image_for_classification(image: Image.Image) -> torch.Tensor:
    """Preprocess image for classification exactly as in the notebooks"""
    try:
        # Use the project's image processing
        image_data = ImageData()
        
        # Resize to standard size
        image = image.resize((224, 224))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize using ImageNet stats (same as the project uses)
        image_array = image_array.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array(image_data.image_net_mean).reshape(1, 1, 3)
        std = np.array(image_data.image_net_std).reshape(1, 1, 3)
        normalized = (image_array - mean) / std
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(normalized).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 224, 224]
        
        return image_tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise e

def classify_image(image: Image.Image, model, language='en') -> dict:
    """Perform classification using the project's model exactly as in notebooks"""
    if model is None:
        return {"error": "Classification model not loaded"}
    
    try:
        # Preprocess image
        image_tensor = preprocess_image_for_classification(image)
        
        with torch.no_grad():
            # Forward pass exactly as in notebooks
            logits = model(image_tensor)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=1)
            
            # Get predicted class
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        
        # Use appropriate language labels
        labels = CLASS_LABELS_ES if language == 'es' else CLASS_LABELS
        
        results = {
            "predicted_class": labels[predicted_class],
            "confidence": round(confidence, 4),
            "all_predictions": [
                {"class": labels[idx.item()], "confidence": round(prob.item(), 4)}
                for prob, idx in zip(top3_probs, top3_indices)
            ]
        }
        return results
        
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        return {"error": str(e)}

def detect_teeth(image: Image.Image, models, score_threshold: float = 0.30) -> dict:
    """Detect teeth using ONLY the project's Detectron2 model."""
    try:
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if models.get("detector_type") == "detectron2" and models.get("detector") is not None:
            outputs = models["detector"](cv_image)
            inst = outputs["instances"].to("cpu")
            boxes = inst.pred_boxes.tensor.numpy().astype(int) if inst.has("pred_boxes") else np.empty((0, 4))
            scores = inst.scores.numpy() if inst.has("scores") else np.array([])

            detections = []
            for (x1, y1, x2, y2), s in zip(boxes, scores):
                if s >= score_threshold:
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(s),
                        "class": "tooth"
                    })
            return {"detections": detections, "num_detections": len(detections), "model_type": "detectron2"}
        else:
            # No fallback - only use proper models
            return {"error": "Detectron2 detection model not available. Please ensure the model is properly trained and loaded."}

    except Exception as e:
        return {"error": f"Detection error: {str(e)}"}

def classify_crops(pil_image: Image.Image, detections: list, model, language='en') -> list:
    """Run classification on each detected tooth crop; attaches results to detections."""
    if model is None or len(detections) == 0:
        return detections
    out = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        # Guard against invalid boxes
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
        crop = pil_image.crop((x1, y1, x2, y2))
        res = classify_image(crop, model, language)
        if "predicted_class" in res:
            d["predicted_class"] = res["predicted_class"]
            d["class_confidence"] = float(res["confidence"])
            d["class_top3"] = res.get("all_predictions", [])
        out.append(d)
    return out

def create_annotated_image(image: Image.Image,
                           classification_result,
                           detection_result,
                           language='en') -> Image.Image:
    """Draw boxes and labels; include per-tooth class if available."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 16)  # Smaller font
    except:
        font = ImageFont.load_default()

    # Global (whole-image) classification (optional)
    if classification_result and "predicted_class" in classification_result:
        txt = f"Image class: {classification_result['predicted_class']} ({classification_result['confidence']:.2%})"
        draw.text((10, 10), txt, fill="red", font=font)

    # Per tooth
    if detection_result and "detections" in detection_result:
        for i, det in enumerate(detection_result["detections"]):
            x1, y1, x2, y2 = det["bbox"]
            
            # Get color based on classification
            predicted_class = det.get('predicted_class', '')
            box_color = CLASSIFICATION_COLORS.get(predicted_class, "green")
            text_color = CLASSIFICATION_COLORS.get(predicted_class, "green")
            
            # Draw thinner bounding box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)  # Thinner width
            
            # Create smaller, more compact label
            label = f"T{i+1}"
            if predicted_class:
                label += f" {predicted_class}"
            if det.get("class_confidence") is not None:
                label += f"({det['class_confidence']:.0%})"
            
            # Position text above the box with smaller offset
            text_y = max(0, y1 - 18)  # Smaller offset
            draw.text((x1, text_y), label.strip(), fill=text_color, font=font)

    return annotated

def main():
    """Main Streamlit app - ONLY using proper project models"""
    st.set_page_config(
        page_title="Dental X-ray Analysis",
        page_icon="🦷",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Language selector in sidebar
    with st.sidebar:
        st.header("🌐 Language / Idioma")
        language = st.selectbox(
            "Language / Idioma:",
            ["English", "Español"],
            key="language_selector"
        )
        
        # Set language code
        lang_code = 'es' if language == "Español" else 'en'
        t = TRANSLATIONS[lang_code]
        
        st.markdown("---")
        st.markdown("### 📊 Model Status")
        
        # Load models
        with st.spinner(t['loading_models']):
            models = load_models()
        
        # Show model status
        if 'classification' in models:
            st.success("✅ Classification Model")
        else:
            st.error("❌ Classification Model")
            
        if 'detector' in models:
            st.success("✅ Detection Model")
        else:
            st.error("❌ Detection Model")
    
    # Main content
    st.title(t['title'])
    st.markdown(t['subtitle'])
    
    # Check if models are loaded
    if not models or ('classification' not in models and 'detector' not in models):
        st.error(t['models_not_loaded'])
        st.stop()
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📁 Upload")
        # File upload
        uploaded_file = st.file_uploader(
            t['file_uploader'],
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file)
            
            # Display original image
            st.markdown("### 📷 " + t['original_image'])
            st.image(image, caption=t['uploaded_xray'], use_column_width=True)
    
    with col2:
        st.markdown("### ⚙️ Settings")
        # Analysis type selection
        analysis_type = st.selectbox(
            t['analysis_type_label'],
            [t['detection_only'], t['classification_only'], t['both']]
        )
        
        # Confidence threshold for detection
        confidence_threshold = st.slider(
            t['confidence_threshold'],
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1
        )
    
    # Results section
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("## 📊 " + t['analysis_results'])
        
        # Create results columns
        results_col1, results_col2 = st.columns([1, 1])
        
        classification_result = None
        detection_result = None
        
        # Classification
        if (t['classification_only'] in analysis_type or t['both'] in analysis_type) and 'classification' in models and models['classification'] is not None:
            with results_col1:
                with st.spinner(t['performing_classification']):
                    classification_result = classify_image(image, models['classification'], lang_code)
                
                if "error" not in classification_result:
                    st.success(f"**{t['classification']}**: {classification_result['predicted_class']}")
                    st.metric(t['confidence'], f"{classification_result['confidence']:.2%}")
                    
                    st.write(f"**{t['all_predictions']}:**")
                    for pred in classification_result['all_predictions']:
                        st.write(f"• {pred['class']}: {pred['confidence']:.2%}")
                else:
                    st.error(f"{t['classification_error']}: {classification_result['error']}")
        
        # Detection
        if (t['detection_only'] in analysis_type or t['both'] in analysis_type) and 'detector' in models and models['detector'] is not None:
            with results_col2:
                with st.spinner(t['performing_detection']):
                    detection_result = detect_teeth(image, models, score_threshold=confidence_threshold)
                
                if "error" not in detection_result:
                    st.success(f"**{t['detection']}**: {detection_result['num_detections']} {t['teeth_found']}")
                    st.write(f"{t['model_type']}: {detection_result['model_type']}")
                    
                    if detection_result['detections']:
                        st.write(f"**{t['top_detections']}:**")
                        for i, detection in enumerate(detection_result['detections'][:5]):
                            bbox = detection['bbox']
                            conf = detection['confidence']
                            class_info = f" ({detection.get('predicted_class', '')})" if detection.get('predicted_class') else ""
                            st.write(f"• {t['tooth']} {i+1}: {t['bbox']}={bbox}, {t['conf']}={conf:.3f}{class_info}")
                else:
                    st.error(f"{t['detection_error']}: {detection_result['error']}")
        
        # Per-tooth classification if both are selected
        if analysis_type == t['both'] and \
           'classification' in models and models['classification'] is not None and \
           detection_result is not None and "error" not in detection_result and detection_result.get('detections'):
            
            st.markdown("---")
            with st.spinner(t['classifying_teeth']):
                detection_result["detections"] = classify_crops(
                    image, detection_result["detections"], models["classification"], lang_code
                )
            
            st.info("✅ " + t['per_tooth_completed'])
        
        # Create and display annotated image
        if detection_result is not None and "error" not in detection_result and detection_result.get('detections'):
            st.markdown("---")
            st.markdown("## 🖼️ " + t['annotated_results'])
            annotated_image = create_annotated_image(image, classification_result, detection_result, lang_code)
            st.image(annotated_image, caption=t['annotated_xray'], use_column_width=True)

if __name__ == "__main__":
    main()
