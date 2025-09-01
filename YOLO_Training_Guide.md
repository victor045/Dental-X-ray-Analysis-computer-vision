# 🦷 YOLO Tooth Detection Training Guide

## 🎯 **Objective**

Train a YOLO model that produces **identical or very similar results** to the existing Detectron2 tooth detection model, enabling seamless replacement in the Streamlit Cloud app.

## 📊 **Current Status Analysis**

### **Detectron2 Model (Reference)**
- **Architecture**: Faster R-CNN with ResNet-101 FPN backbone
- **Training Data**: 634 dental X-ray images with COCO annotations
- **Classes**: 1 (tooth)
- **Output**: 28-31 detections per image with high confidence (0.7-0.9)
- **Performance**: mAP50 ~0.85, mAP50-95 ~0.65

### **YOLO Model (Target)**
- **Architecture**: YOLOv8 with CSP backbone
- **Training Data**: Same 634 images converted to YOLO format
- **Classes**: 1 (tooth)
- **Target Output**: Same detection count and confidence distribution
- **Target Performance**: Match Detectron2 metrics

## 🚀 **Complete Training Pipeline**

### **Step 1: Data Preparation**

```bash
# Run the complete training pipeline
python3 train_yolo_tooth_detection.py
```

This script automatically:
1. **Downloads** the detection dataset (634 images)
2. **Creates** train/val/test splits (same as Detectron2)
3. **Converts** COCO annotations to YOLO format
4. **Trains** YOLO model with optimal parameters
5. **Validates** model performance
6. **Compares** results with Detectron2

### **Step 2: Data Format Conversion**

#### **COCO to YOLO Conversion**

**COCO Format** (Detectron2):
```json
{
  "bbox": [x, y, width, height],  // Absolute coordinates
  "category_id": 0,
  "area": 1234,
  "iscrowd": 0
}
```

**YOLO Format** (Training):
```txt
0 0.123456 0.234567 0.045678 0.056789
```
- `0`: Class ID (tooth)
- `0.123456`: Normalized center X
- `0.234567`: Normalized center Y  
- `0.045678`: Normalized width
- `0.056789`: Normalized height

#### **Conversion Formula**
```python
# COCO: [x, y, width, height]
x, y, w, h = bbox

# YOLO: [class, x_center, y_center, width, height] (normalized)
x_center = (x + w/2) / img_width
y_center = (y + h/2) / img_height
width = w / img_width
height = h / img_height
```

### **Step 3: YOLO Training Configuration**

#### **Model Architecture**
```python
model = YOLO('yolov8n.pt')  # Start with nano model
```

#### **Training Parameters**
```python
training_args = {
    'data': 'dataset.yaml',
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'device': 'cpu',  # CPU for consistency
    'lr0': 0.01,      # Initial learning rate
    'lrf': 0.1,       # Final learning rate factor
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'patience': 20,   # Early stopping
    'save_period': 10,
    'box': 7.5,       # Box loss gain
    'cls': 0.5,       # Class loss gain
    'dfl': 1.5,       # DFL loss gain
}
```

#### **Dataset Configuration**
```yaml
# dataset.yaml
path: /path/to/yolo_dataset
train: train/images
val: val/images
test: test/images
nc: 1
names: ['tooth']
```

### **Step 4: Training Process**

#### **Data Splits** (Same as Detectron2)
- **Train**: ~534 images (84%)
- **Validation**: ~50 images (8%)
- **Test**: ~50 images (8%)

#### **Training Schedule**
1. **Warmup**: 3 epochs with increasing learning rate
2. **Main Training**: 97 epochs with cosine annealing
3. **Early Stopping**: If no improvement for 20 epochs
4. **Checkpointing**: Save every 10 epochs

#### **Loss Functions**
- **Box Loss**: CIoU loss for bounding box regression
- **Class Loss**: BCE loss for classification
- **DFL Loss**: Distribution Focal Loss for regression

### **Step 5: Model Validation**

#### **Metrics to Match**
- **mAP50**: Should be > 0.80 (target: 0.85)
- **mAP50-95**: Should be > 0.60 (target: 0.65)
- **Precision**: Should be > 0.85
- **Recall**: Should be > 0.80

#### **Detection Count Analysis**
- **Target**: 28-31 detections per image
- **Confidence Distribution**: 0.7-0.9 range
- **IoU Threshold**: 0.5 for matching

### **Step 6: Comparison Testing**

#### **Automated Comparison**
```bash
python3 test_detection_comparison.py
```

This script:
1. **Loads** both Detectron2 and YOLO models
2. **Runs** inference on same test images
3. **Calculates** IoU between detections
4. **Generates** side-by-side visualizations
5. **Reports** matching statistics

#### **Success Criteria**
- **IoU > 0.7** for matched detections
- **Detection count within ±20%** of Detectron2
- **Confidence correlation > 0.8**
- **False positive rate < 10%**

## 🔧 **Optimization Strategies**

### **1. Data Augmentation**
```python
# YOLO built-in augmentations
augmentations = {
    'hsv_h': 0.015,    # HSV-Hue augmentation
    'hsv_s': 0.7,      # HSV-Saturation augmentation
    'hsv_v': 0.4,      # HSV-Value augmentation
    'degrees': 0.0,    # Image rotation
    'translate': 0.1,  # Image translation
    'scale': 0.5,      # Image scaling
    'shear': 0.0,      # Image shear
    'perspective': 0.0, # Perspective transform
    'flipud': 0.0,     # Vertical flip
    'fliplr': 0.5,     # Horizontal flip
    'mosaic': 1.0,     # Mosaic augmentation
    'mixup': 0.0,      # Mixup augmentation
}
```

### **2. Model Architecture Tuning**
```python
# Try different YOLO variants
models = [
    'yolov8n.pt',  # Nano (fastest)
    'yolov8s.pt',  # Small
    'yolov8m.pt',  # Medium
    'yolov8l.pt',  # Large
    'yolov8x.pt',  # Extra large (best)
]
```

### **3. Hyperparameter Optimization**
```python
# Grid search for optimal parameters
param_grid = {
    'lr0': [0.01, 0.005, 0.02],
    'lrf': [0.1, 0.05, 0.2],
    'momentum': [0.937, 0.9, 0.95],
    'weight_decay': [0.0005, 0.0001, 0.001],
    'box': [7.5, 5.0, 10.0],
    'cls': [0.5, 0.3, 0.7],
}
```

## 📈 **Expected Results**

### **Training Progress**
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/100     0G       2.123      1.456      1.234         45        640: 100%|██████████| 34/34 [00:45<00:00,  1.32it/s]
  2/100     0G       1.987      1.234      1.123         45        640: 100%|██████████| 34/34 [00:44<00:00,  1.31it/s]
  ...
 99/100     0G       0.234      0.123      0.234         45        640: 100%|██████████| 34/34 [00:43<00:00,  1.30it/s]
100/100     0G       0.223      0.112      0.223         45        640: 100%|██████████| 34/34 [00:43<00:00,  1.30it/s]
```

### **Validation Results**
```
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 7/7 [00:02<00:00,  3.12it/s]
                   all         50        234      0.856      0.823      0.847      0.652
                 tooth         50        234      0.856      0.823      0.847      0.652
```

### **Comparison Results**
```
📊 Detection Comparison:
Detectron2: 28 detections
YOLO: 29 detections

🔍 Matching Results:
Matched detections: 25
Unmatched Detectron2: 3
Unmatched YOLO: 4
Average IoU: 0.823
Average confidence difference: 0.045
```

## 🚀 **Integration with Streamlit Cloud**

### **Model Loading**
```python
# In streamlit_cloud_app.py
def load_models():
    # Load trained YOLO model
    yolo_model_path = os.path.join(model_dir, "YOLO_Toothdetector", "version_1", "tooth_yolo.pt")
    if os.path.exists(yolo_model_path):
        yolo_model = YOLO(yolo_model_path)
        yolo_model.to('cpu')
        return yolo_model
```

### **Inference**
```python
def detect_teeth(image, models, score_threshold=0.30):
    if models.get("detector_type") == "yolo":
        res = models["detector"](image, verbose=False)[0]
        detections = []
        for b, s, c in zip(res.boxes.xyxy.cpu().numpy(),
                          res.boxes.conf.cpu().numpy(),
                          res.boxes.cls.cpu().numpy()):
            if s >= score_threshold:
                detections.append({
                    "bbox": b.tolist(),
                    "confidence": float(s),
                    "class": "tooth"
                })
        return {"detections": detections, "num_detections": len(detections), "model_type": "yolo"}
```

## 🎯 **Success Metrics**

### **Primary Goals**
- ✅ **Detection Count**: Match Detectron2 (28-31 detections)
- ✅ **Confidence Distribution**: Similar to Detectron2 (0.7-0.9)
- ✅ **Bounding Box Accuracy**: IoU > 0.7 for matches
- ✅ **Performance**: mAP50 > 0.80

### **Secondary Goals**
- ✅ **Inference Speed**: Faster than Detectron2
- ✅ **Memory Usage**: Lower than Detectron2
- ✅ **Model Size**: Smaller than Detectron2
- ✅ **Cloud Compatibility**: Works on Streamlit Cloud

## 🔄 **Iterative Improvement**

### **Phase 1: Basic Training**
- Train YOLOv8n with default parameters
- Validate against Detectron2
- Identify gaps in performance

### **Phase 2: Parameter Tuning**
- Optimize learning rate and schedule
- Adjust loss function weights
- Fine-tune augmentation parameters

### **Phase 3: Architecture Optimization**
- Try larger YOLO models (s, m, l, x)
- Experiment with different backbones
- Test ensemble methods

### **Phase 4: Final Validation**
- Comprehensive comparison testing
- Performance benchmarking
- Cloud deployment testing

## 📝 **Troubleshooting**

### **Common Issues**

1. **Low Detection Count**
   - Increase confidence threshold
   - Adjust loss function weights
   - Add more training epochs

2. **Poor IoU Matching**
   - Check annotation conversion
   - Verify coordinate normalization
   - Review data augmentation

3. **High False Positives**
   - Increase classification loss weight
   - Add negative samples
   - Adjust NMS parameters

4. **Training Instability**
   - Reduce learning rate
   - Increase batch size
   - Add gradient clipping

### **Debug Commands**
```bash
# Check dataset structure
ls -la data/dentex/dentex_detection/yolo_dataset/

# Validate annotations
python3 -c "from ultralytics import YOLO; YOLO('dataset.yaml').val()"

# Test inference
python3 test_detection_comparison.py

# Monitor training
tensorboard --logdir runs/detect/train
```

---

**🎉 Goal**: Train a YOLO model that produces **identical results** to Detectron2, enabling seamless replacement in the Streamlit Cloud app while maintaining all functionality and performance characteristics.


