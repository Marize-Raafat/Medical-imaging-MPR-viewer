# Installation Guide

## Step 1: Install Python

1. Go to https://www.python.org/downloads/
2. Download Python 3.8 or higher
3. Run the installer
4. **IMPORTANT**: Check the box "Add Python to PATH"
5. Click "Install Now"

Verify installation:
```bash
python --version
```

## Step 2: Clone This Repository
```bash
git clone https://github.com/Marize-Raafat/Medical-imaging-MPR-viewer.git
cd mpr-viewer
```

## Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This will take 5-10 minutes.

## Step 4: Run the Application
```bash
python src/MPR_viewer_final.py
```

## Troubleshooting

### "python not recognized"
- Reinstall Python and check "Add to PATH"

### TensorFlow issues
- Try: `pip install tensorflow-cpu` (CPU-only version)

### Out of memory
- Close other applications
- Use smaller image files