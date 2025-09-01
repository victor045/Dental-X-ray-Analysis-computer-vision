#!/usr/bin/env python3
"""
Train YOLO model for tooth detection to match Detectron2 output
This script follows the same data preparation and training pipeline as Detectron2
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import yaml

# Add src to Python path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Set DATA_ROOT environment variable
os.environ['DATA_ROOT'] = os.path.join(os.getcwd(), 'data')

# Import this module with autoreload
import dentexmodel as dm
from dentexmodel.dentexdata import DentexData, val_test_split
from dentexmodel.fileutils import FileOP
from dentexmodel.imageproc import ImageData

def download_and_prepare_data():
    """Download and prepare the detection dataset"""
    print("🔄 Downloading and preparing detection dataset...")
    
    data_root = os.environ.get("DATA_ROOT", "./data")
    dentex_dir = os.path.join(data_root, 'dentex')
    data_dir = os.path.join(dentex_dir, 'dentex_detection')
    
    # Create directories
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Download detection dataset
    dtx = DentexData(data_dir=data_dir)
    url = dtx.detection_url
    print(f"📥 Downloading from: {url}")
    
    data_tar_file = dtx.download_image_data(url=url)
    
    # Check if download was successful
    image_dir = os.path.join(data_dir, 'quadrant_enumeration', 'xrays')
    if not os.path.exists(image_dir):
        print("❌ Image directory not found after download")
        return False
    
    print("✅ Dataset downloaded successfully")
    return True

def create_data_splits():
    """Create train/val/test splits"""
    print("🔄 Creating data splits...")
    
    data_root = os.environ.get("DATA_ROOT", "./data")
    data_dir = os.path.join(data_root, 'dentex', 'dentex_detection')
    image_dir = os.path.join(data_dir, 'quadrant_enumeration', 'xrays')
    
    # Check if initial dataset parquet file exists, if not create it
    data_df_file = os.path.join(data_dir, 'dentex_detection_dataset.parquet')
    if not os.path.exists(data_df_file):
        print("📝 Creating initial dataset parquet file...")
        
        # Get list of image files
        import glob
        file_list = glob.glob(os.path.join(image_dir, '*.png'))
        if not file_list:
            print("❌ No image files found. Check if dataset was extracted properly.")
            return False
        
        print(f"📸 Found {len(file_list)} images")
        
        # Create files dataframe
        file_name_list = [os.path.basename(file) for file in file_list]
        im_number_list = [int(os.path.splitext(file)[0].rsplit('_', maxsplit=1)[-1]) for file in file_name_list]
        files = pd.DataFrame({
            'image_number': im_number_list,
            'file_name': file_name_list,
            'file_path': file_list
        }).sort_values(by='image_number', ascending=True).reset_index(drop=True)
        
        # Load annotations
        annotation_file = os.path.join(data_dir, 'quadrant_enumeration', 'train_quadrant_enumeration.json')
        dtx = DentexData(data_dir=data_dir)
        annotations = dtx.load_annotations(annotation_file)
        
        # Add image ids to the files dataframe
        js_im_df = pd.DataFrame(annotations.get('images')).\
                    merge(files, on='file_name', how='inner').\
                    sort_values(by='id', ascending=True).\
                    reset_index(drop=True).\
                    rename(columns={'id': 'image_id'}).\
                    drop(['height', 'width'], axis=1)
        
        # Save the initial dataset
        js_im_df.to_parquet(data_df_file)
        print(f"✅ Initial dataset saved: {data_df_file}")
    
    # Now load the data frame and create splits
    data_df = pd.read_parquet(data_df_file)
    dtx = DentexData(data_dir=data_dir)
    
    # Create data splits
    dset_df = val_test_split(data=data_df, n_test_per_class=50, n_val_per_class=50)
    
    # Save the data split
    datasplit_file = os.path.join(data_dir, 'dentex_detection_datasplit.parquet')
    dset_df.to_parquet(datasplit_file)
    
    print(f"✅ Data splits created: {len(dset_df)} total images")
    print(f"   Train: {len(dset_df[dset_df['dataset'] == 'train'])}")
    print(f"   Val: {len(dset_df[dset_df['dataset'] == 'val'])}")
    print(f"   Test: {len(dset_df[dset_df['dataset'] == 'test'])}")
    
    return True

def create_yolo_annotations():
    """Convert COCO annotations to YOLO format"""
    print("🔄 Creating YOLO annotations...")
    
    data_root = os.environ.get("DATA_ROOT", "./data")
    data_dir = os.path.join(data_root, 'dentex', 'dentex_detection')
    image_dir = os.path.join(data_dir, 'quadrant_enumeration', 'xrays')
    
    # Load data splits
    datasplit_file = os.path.join(data_dir, 'dentex_detection_datasplit.parquet')
    data_df = pd.read_parquet(datasplit_file)
    
    # Load original annotations
    annotation_file = os.path.join(data_dir, 'quadrant_enumeration', 'train_quadrant_enumeration.json')
    dtx = DentexData(data_dir=data_dir)
    annotations = dtx.load_annotations(annotation_file)
    
    # Create YOLO dataset structure
    yolo_dataset_dir = os.path.join(data_dir, 'yolo_dataset')
    Path(yolo_dataset_dir).mkdir(parents=True, exist_ok=True)
    
    # Create directories for each split
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(yolo_dataset_dir, split)
        Path(split_dir).mkdir(exist_ok=True)
        Path(os.path.join(split_dir, 'images')).mkdir(exist_ok=True)
        Path(os.path.join(split_dir, 'labels')).mkdir(exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"📝 Processing {split} split...")
        split_df = data_df[data_df['dataset'] == split]
        
        for idx, row in split_df.iterrows():
            image_file = row['file_path']
            image_name = row['file_name']
            image_id = row['image_id']
            
            # Copy image to YOLO dataset
            dest_image = os.path.join(yolo_dataset_dir, split, 'images', image_name)
            shutil.copy2(image_file, dest_image)
            
            # Create YOLO annotation
            yolo_annotation = create_yolo_annotation(annotations, image_id, image_file)
            
            # Save YOLO annotation
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_file = os.path.join(yolo_dataset_dir, split, 'labels', label_name)
            
            with open(label_file, 'w') as f:
                for annotation in yolo_annotation:
                    f.write(f"{annotation}\n")
    
    # Create YOLO dataset config
    create_yolo_dataset_yaml(yolo_dataset_dir)
    
    print("✅ YOLO annotations created successfully")
    return True

def create_yolo_annotation(annotations, image_id, image_file):
    """Convert COCO annotation to YOLO format for a single image"""
    # Load image to get dimensions
    img = ImageData().load_image(image_file)
    img_height, img_width = img.shape[:2]
    
    # Find annotations for this image
    image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
    
    yolo_annotations = []
    for ann in image_annotations:
        # COCO format: [x, y, width, height]
        # YOLO format: [class_id, x_center, y_center, width, height] (normalized)
        
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, w, h = bbox
        
        # Convert to YOLO format (normalized coordinates)
        x_center = (x + w/2) / img_width
        y_center = (y + h/2) / img_height
        width = w / img_width
        height = h / img_height
        
        # Class ID (0 for tooth)
        class_id = 0
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations

def create_yolo_dataset_yaml(dataset_dir):
    """Create YOLO dataset configuration file"""
    yaml_content = {
        'path': dataset_dir,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,  # number of classes
        'names': ['tooth']  # class names
    }
    
    yaml_file = os.path.join(dataset_dir, 'dataset.yaml')
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"✅ YOLO dataset config created: {yaml_file}")

def train_yolo_model():
    """Train YOLO model"""
    print("🔄 Training YOLO model...")
    
    try:
        from ultralytics import YOLO
        
        data_root = os.environ.get("DATA_ROOT", "./data")
        data_dir = os.path.join(data_root, 'dentex', 'dentex_detection')
        yolo_dataset_dir = os.path.join(data_dir, 'yolo_dataset')
        dataset_yaml = os.path.join(yolo_dataset_dir, 'dataset.yaml')
        
        # Model output directory
        model_dir = os.path.join(data_root, 'model')
        yolo_output_dir = os.path.join(model_dir, 'YOLO_Toothdetector', 'version_1')
        Path(yolo_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize YOLO model
        model = YOLO('yolov8n.pt')  # Start with nano model
        
        # Training configuration optimized for limited GPU memory
        training_args = {
            'data': dataset_yaml,
            'epochs': 100,
            'imgsz': 640,
            'batch': 4,  # Reduced batch size for GPU memory
            'device': 0,  # Use GPU (device 0) for faster training
            'project': yolo_output_dir,
            'name': 'yolo_tooth_detection',
            'save_period': 10,
            'patience': 20,
            'lr0': 0.01,
            'lrf': 0.1,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'workers': 2,  # Reduced workers for GPU memory
            'cache': False,  # Disable caching to save memory
            'amp': False  # Disable AMP to avoid CUDA issues
        }
        
        # Train the model
        results = model.train(**training_args)
        
        # Save the final model
        final_model_path = os.path.join(yolo_output_dir, 'tooth_yolo.pt')
        model.save(final_model_path)
        
        print(f"✅ YOLO model trained successfully")
        print(f"📁 Model saved to: {final_model_path}")
        
        return final_model_path
        
    except Exception as e:
        print(f"❌ Error training YOLO model: {e}")
        return None

def validate_yolo_model(model_path):
    """Validate YOLO model against test set"""
    print("🔄 Validating YOLO model...")
    
    try:
        from ultralytics import YOLO
        
        # Load the trained model
        model = YOLO(model_path)
        
        data_root = os.environ.get("DATA_ROOT", "./data")
        data_dir = os.path.join(data_root, 'dentex', 'dentex_detection')
        yolo_dataset_dir = os.path.join(data_dir, 'yolo_dataset')
        dataset_yaml = os.path.join(yolo_dataset_dir, 'dataset.yaml')
        
        # Validate on test set
        results = model.val(data=dataset_yaml, split='test')
        
        print("✅ YOLO model validation completed")
        print(f"📊 mAP50: {results.box.map50:.3f}")
        print(f"📊 mAP50-95: {results.box.map:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error validating YOLO model: {e}")
        return None

def compare_with_detectron2():
    """Compare YOLO results with Detectron2"""
    print("🔄 Comparing YOLO with Detectron2...")
    
    # This will use the comparison script we created earlier
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'test_detection_comparison.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Comparison completed successfully")
            print("📊 Check the generated comparison results")
        else:
            print(f"❌ Comparison failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error running comparison: {e}")

def main():
    """Main training pipeline"""
    print("🦷 YOLO Tooth Detection Training Pipeline")
    print("=" * 60)
    
    # Step 1: Download and prepare data
    if not download_and_prepare_data():
        print("❌ Failed to download data")
        return False
    
    # Step 2: Create data splits
    if not create_data_splits():
        print("❌ Failed to create data splits")
        return False
    
    # Step 3: Create YOLO annotations
    if not create_yolo_annotations():
        print("❌ Failed to create YOLO annotations")
        return False
    
    # Step 4: Train YOLO model
    model_path = train_yolo_model()
    if not model_path:
        print("❌ Failed to train YOLO model")
        return False
    
    # Step 5: Validate model
    validation_results = validate_yolo_model(model_path)
    if not validation_results:
        print("❌ Failed to validate model")
        return False
    
    # Step 6: Compare with Detectron2
    compare_with_detectron2()
    
    print("\n🎉 YOLO training pipeline completed successfully!")
    print(f"📁 Model saved to: {model_path}")
    print("🚀 Ready to use in Streamlit Cloud app!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
