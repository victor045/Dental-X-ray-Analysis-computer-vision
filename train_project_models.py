#!/usr/bin/env python3
"""
Train the Dentex models using the original project structure
This script follows the complete pipeline from the notebooks
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"\n🔄 {description}")
    print("=" * 50)
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.join(os.getcwd(), 'src') + ':' + env.get('PYTHONPATH', '')
        
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.getcwd(),
                              env=env)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print(f"❌ {description} failed")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False
    
    return True

def check_model_exists():
    """Check if trained models already exist"""
    data_root = os.environ.get('DATA_ROOT', './data')
    model_dir = os.path.join(data_root, 'model')
    
    # Check for classification model
    classification_checkpoint = os.path.join(model_dir, 'FancyLR', 'version_1', 'checkpoints', 'last.ckpt')
    
    if os.path.exists(classification_checkpoint):
        print(f"✅ Classification model found: {classification_checkpoint}")
        return True
    else:
        print(f"❌ Classification model not found: {classification_checkpoint}")
        return False

def main():
    """Main training pipeline"""
    print("🦷 Dentex Model Training Pipeline")
    print("=" * 60)
    
    # Check if models already exist
    if check_model_exists():
        print("\n🎉 Models already trained! Skipping training.")
        return True
    
    # Step 1: Download and prepare dataset
    if not run_script("notebooks/classification/01_dentex_download.py", "Downloading dataset"):
        return False
    
    # Step 2: Create cropped dataset
    if not run_script("notebooks/classification/02_create_dataset.py", "Creating cropped dataset"):
        return False
    
    # Step 3: Create train/val/test split
    if not run_script("notebooks/classification/03_train_val_test_split.py", "Creating data splits"):
        return False
    
    # Step 4: Train the classification model
    if not run_script("notebooks/classification/07_model_training_fancy.py", "Training classification model"):
        return False
    
    # Step 5: Verify model was created
    if check_model_exists():
        print("\n🎉 Model training completed successfully!")
        return True
    else:
        print("\n❌ Model training failed - model not found")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready to use the trained models!")
        print("You can now run the Streamlit app with:")
        print("streamlit run streamlit_app.py")
    else:
        print("\n❌ Training failed. Please check the errors above.")
        sys.exit(1)
