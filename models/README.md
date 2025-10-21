# AI Model Files

## Required Files

This directory should contain:
- `config.json` - Keras model architecture
- `model.weights.h5` - Trained model weights

## Getting the Model

The model files are not included in this repository due to size.

### If you have trained your own model:
1. Place `config.json` in this folder
2. Place `model.weights.h5` in this folder

### If you need the pre-trained model:
Contact the repository owner or see the Releases page.

## Model Specifications

- **Input**: 3D volume (128×128×128×1)
- **Output**: 3-class classification (Axial, Coronal, Sagittal)
- **Framework**: TensorFlow/Keras
- **Architecture**: 3D Convolutional Neural Network