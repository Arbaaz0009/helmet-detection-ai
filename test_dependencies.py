#!/usr/bin/env python3
"""
Test script to verify all dependencies can be imported
"""

def test_imports():
    """Test importing all required packages"""
    print("🔍 Testing package imports...")
    
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
    except ImportError as e:
        print(f"❌ streamlit import failed: {e}")
    
    try:
        import cv2
        print("✅ opencv-python-headless imported successfully")
    except ImportError as e:
        print(f"❌ opencv import failed: {e}")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
    
    try:
        from PIL import Image
        print("✅ pillow imported successfully")
    except ImportError as e:
        print(f"❌ pillow import failed: {e}")
    
    try:
        import torch
        print("✅ torch imported successfully")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
    
    try:
        from ultralytics import YOLO
        print("✅ ultralytics imported successfully")
    except ImportError as e:
        print(f"❌ ultralytics import failed: {e}")
    
    print("\n🎉 All imports completed!")

if __name__ == "__main__":
    test_imports() 