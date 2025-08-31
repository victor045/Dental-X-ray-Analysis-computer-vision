#!/bin/bash

# Dental Image Analysis Web Application Startup Script

echo "🦷 Dental Image Analysis Web Application"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: Please run this script from the web_app directory"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import flask, torch, PIL, cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Check if model exists
if [ ! -f "../data/model/classification_model.pth" ]; then
    echo "⚠️  Warning: No trained model found"
    echo "📝 Creating a dummy model for demonstration..."
    python3 train_model.py
fi

# Start the application
echo "🚀 Starting the web application..."
echo "🌐 Access the application at: http://localhost:5000"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

python3 app.py

