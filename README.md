# Professional MPR Viewer with AI Orientation Prediction

A medical imaging application for Multi-Planar Reconstruction (MPR) with AI-powered orientation detection.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Features

- **4 Synchronized Views**: Axial, Sagittal, Coronal, and Oblique planes
- **Interactive Navigation**: Drag crosshairs to navigate through 3D volumes
- **AI Orientation Prediction**: Automatically detect view orientation
- **Organ Detection**: Identify anatomical regions from DICOM metadata
- **Surface Boundary Mode**: Display 2D contours from segmentation masks
- **ROI Export**: Save regions of interest as NIfTI files

## 🚀 Quick Start

### Installation
```bash
# Clone this repository
git clone https://github.com/Marize-Raafat/mpr-viewer.git
cd mpr-viewer

# Install dependencies
pip install -r requirements.txt

# Run the application
python src/MPR_viewer_final.py
```

## 📖 Usage

1. Click **"Load NIfTI"** or **"Load DICOM"** to open a medical image
2. Drag the colored crosshairs to navigate
3. Use **Oblique Tilt** sliders to adjust the oblique plane
4. Click **"Predict Orientation"** to run AI classification
5. Click **"Detect Organ"** (DICOM only) to identify anatomy

## 🤖 AI Model

The orientation prediction model requires:
- `models/config.json` - Model architecture
- `models/model.weights.h5` - Trained weights

**Note**: Model files not included due to size.

## 🛠️ Requirements

- Python 3.8+
- 8GB RAM minimum
- GPU optional (CPU inference supported)

## 📁 Project Structure
```
mpr-viewer/
├── src/                    # Source code
├── models/                 # AI model files
├── docs/                   # Documentation
└── requirements.txt        # Python dependencies
```

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Marize Raafat**
- GitHub: [@Marize-Raafat](https://github.com/Marize-Raafat)
- Email: marizeraafat020@gmail.com