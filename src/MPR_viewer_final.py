"""
Professional MPR Viewer - Full 4-view program with Oblique/Surface Boundary + AI Orientation Prediction
Features:
 - Load NIfTI (.nii/.nii.gz) and single DICOM (.dcm)
 - 4 viewports: Axial, Sagittal, Coronal, Oblique
 - Colored reference lines per view
 - Draggable lines that sync views
 - Oblique plane controls (tilt X/Y)
 - Per-view playback (Play/Pause + step) and slider for slice index
 - ROI export (axial slice range) to NIfTI
 - Surface Boundary mode: show 2D contour/boundary in Oblique viewport
 - AI Orientation Prediction: predict view orientation of loaded volume
 - Writes default QSS file 'dark_red.qss' (dark red theme)
"""
import sys
import os
import math
import traceback
import json
import numpy as np

# required libs
try:
    import nibabel as nib
except Exception:
    print("ERROR: nibabel missing. Install: pip install nibabel")
    sys.exit(1)

try:
    from PyQt5.QtWidgets import (
        QApplication, QComboBox, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QFileDialog, QLabel, QSlider, QMessageBox,
        QGridLayout, QGroupBox, QSpinBox, QDoubleSpinBox
    )
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
except Exception:
    print("ERROR: PyQt5 missing. Install: pip install PyQt5")
    sys.exit(1)

try:
    from scipy.ndimage import map_coordinates, zoom as ndi_zoom
except Exception:
    print("ERROR: scipy missing. Install: pip install scipy")
    sys.exit(1)

# optional: pydicom (for single .dcm file load)
try:
    import pydicom
    DICOM_AVAILABLE = True
except Exception:
    DICOM_AVAILABLE = False

# skimage for contours
try:
    from skimage import measure
except Exception:
    print("ERROR: scikit-image missing. Install: pip install scikit-image")
    sys.exit(1)

# optional: TensorFlow/Keras for orientation model
try:
    from tensorflow.keras.models import model_from_json
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
    model_from_json = None
    tf = None


# ======================== ORIENTATION MODEL LOADER & HELPERS ========================

def load_orientation_model(folder_path="."):
    """
    Load a Keras model from config.json + model.weights.h5 located in folder_path.
    Handles common shapes where config.json may contain nested keys like 'model_config' or 'config'.
    Returns the model or None on failure.
    """
    if not TF_AVAILABLE or model_from_json is None:
        print("TensorFlow / Keras not available. Install tensorflow to use AI features.")
        return None

    cfg_path = os.path.join(folder_path, "config.json")
    weights_path = os.path.join(folder_path, "model.weights.h5")
    if not os.path.exists(cfg_path) or not os.path.exists(weights_path):
        print("Orientation model files not found in folder:", folder_path)
        return None

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Extract model JSON part depending on common formats
        if isinstance(raw, dict):
            if "model_config" in raw:
                model_json = json.dumps(raw["model_config"])
            elif "config" in raw and ("layers" in raw or "class_name" in raw):
                model_json = json.dumps(raw)
            elif "config" in raw:
                model_json = json.dumps(raw["config"])
            else:
                model_json = json.dumps(raw)
        else:
            model_json = json.dumps(raw)

        model = model_from_json(model_json)
        model.load_weights(weights_path)
        # try to compile (not necessary for inference but sometimes helpful)
        try:
            model.compile(optimizer="adam", loss="categorical_crossentropy")
        except Exception:
            pass

        # try infer input shape; fallback to (None,128,128,128,1)
        inp_shape = getattr(model, "input_shape", None)
        if inp_shape is None:
            for layer in getattr(model, "layers", []):
                if hasattr(layer, "input_shape") and layer.input_shape:
                    inp_shape = layer.input_shape
                    break
        if inp_shape is None:
            inp_shape = (None, 128, 128, 128, 1)
            print("Could not detect model input shape; using default (None,128,128,128,1)")

        model._inferred_input_shape = inp_shape
        print("Orientation model loaded. inferred input shape:", model._inferred_input_shape)
        return model

    except Exception as e:
        print("Error loading orientation model:", e)
        return None


def resize_volume_to_target(vol, target_shape):
    """
    Resize a 3D numpy volume to target_shape = (H, W, D) using scipy.ndimage.zoom.
    Volume is normalized using robust 1-99 percentile before resizing and returned as float32.
    """
    vol = np.asarray(vol, dtype=np.float32)
    if vol.size == 0:
        return np.zeros(target_shape, dtype=np.float32)

    # robust normalization
    vmin = np.percentile(vol, 1.0)
    vmax = np.percentile(vol, 99.0)
    if vmax > vmin:
        vol = np.clip(vol, vmin, vmax)
        vol = (vol - vmin) / (vmax - vmin)
    else:
        vol = np.zeros_like(vol)

    src_shape = vol.shape
    if len(src_shape) != 3:
        vol = vol.reshape(src_shape + (1,)) if vol.ndim == 2 else np.squeeze(vol)

    factors = (float(target_shape[0]) / max(1, src_shape[0]),
               float(target_shape[1]) / max(1, src_shape[1]),
               float(target_shape[2]) / max(1, src_shape[2]))
    resized = ndi_zoom(vol, factors, order=1)
    resized = resized.astype(np.float32)

    # if shape mismatch due to rounding, pad/crop
    if resized.shape != tuple(target_shape):
        th, tw, td = target_shape
        padded = np.zeros(target_shape, dtype=np.float32)
        min0 = min(th, resized.shape[0])
        min1 = min(tw, resized.shape[1])
        min2 = min(td, resized.shape[2])
        padded[:min0, :min1, :min2] = resized[:min0, :min1, :min2]
        resized = padded
    return resized

# load model once at import if possible (non-fatal if not found)
orientation_model = load_orientation_model(".")


# ======================== ORIGINAL VIEWER CODE ========================

# Color mapping for views
COLORS = {
    'Axial': '#2A7BD8',    # Blue
    'Coronal': '#E53935',  # Red
    'Sagittal': '#F9A825', # Yellow
    'Oblique': '#2ECC71'   # Green
}


# ---------- Utility clamp ----------
def clamp(v, a, b):
    return max(a, min(b, v))


# ---------- Canvas (QLabel-based) ----------
class ImageCanvas(QLabel):
    """Simple image canvas that displays a 2D numpy array (grayscale)
       with two colored reference lines (v and h) and optionally an oblique projection line.
    """
    def __init__(self, view_name, parent_viewer):
        super().__init__()
        self.view_name = view_name
        self.parent_viewer = parent_viewer
        self.image_data = None
        self.v_pos = 0.0
        self.h_pos = 0.0
        self.zoom = 1.0
        self.dragging = None
        self.setMinimumSize(360, 300)
        self.setStyleSheet("border:1px solid #444; background-color: black;")
        self.setAlignment(Qt.AlignCenter)
        self.active = False

    def set_image(self, arr, v_pos=None, h_pos=None):
        if arr is None or getattr(arr, "size", 0) == 0:
            self.image_data = None
            self.clear()
            return
        img = np.asarray(arr, dtype=np.float32)
        # ensure 2D
        if img.ndim > 2:
            img = np.squeeze(img)
        if img.ndim != 2:
            img = np.zeros((10, 10), dtype=np.float32)
        self.image_data = img
        h, w = img.shape
        if v_pos is None:
            v_pos = w // 2
        if h_pos is None:
            h_pos = h // 2
        self.v_pos = clamp(float(v_pos), 0, w - 1)
        self.h_pos = clamp(float(h_pos), 0, h - 1)
        self.update_display()

    def update_display(self, oblique_direction_2d=None, boundary_contours=None):
        # boundary_contours: list of 2D contour arrays to draw (in image pixel coords)
        if self.image_data is None:
            self.clear()
            return
        try:
            img = self.image_data.copy()
            # Normalize using percentiles to handle CT range
            if img.max() > img.min():
                vmin = np.percentile(img, 1)
                vmax = np.percentile(img, 99)
                if vmax == vmin:
                    arr8 = np.zeros_like(img, dtype=np.uint8)
                else:
                    arr8 = np.clip(img, vmin, vmax)
                    arr8 = ((arr8 - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            else:
                arr8 = np.zeros_like(img, dtype=np.uint8)

            h, w = arr8.shape
            bytes_per_line = w
            qimg = QImage(arr8.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            sw = int(w * self.zoom); sh = int(h * self.zoom)
            if sw <= 0 or sh <= 0:
                return
            qimg = qimg.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pix = QPixmap.fromImage(qimg)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing)

            # choose line colors
            if self.view_name == 'Axial':
                color_v = QColor(COLORS['Sagittal'])
                color_h = QColor(COLORS['Coronal'])
            elif self.view_name == 'Sagittal':
                color_v = QColor(COLORS['Coronal'])
                color_h = QColor(COLORS['Axial'])
            elif self.view_name == 'Coronal':
                color_v = QColor(COLORS['Sagittal'])
                color_h = QColor(COLORS['Axial'])
            else:
                color_v = color_h = QColor('#FFFFFF')

            # vertical
            pen_v = QPen(color_v)
            pen_v.setWidth(2)
            painter.setPen(pen_v)
            cx = int(clamp(self.v_pos * self.zoom, 0, pix.width() - 1))
            painter.drawLine(cx, 0, cx, pix.height())

            # horizontal
            pen_h = QPen(color_h)
            pen_h.setWidth(2)
            painter.setPen(pen_h)
            cy = int(clamp(self.h_pos * self.zoom, 0, pix.height() - 1))
            painter.drawLine(0, cy, pix.width(), cy)

            # small color square top-right indicating view
            square_w, square_h = 18, 12
            margin = 6
            painter.fillRect(pix.width() - square_w - margin, margin, square_w, square_h, QColor(COLORS.get(self.view_name, '#FFFFFF')))

            # draw oblique projection line if provided
            if oblique_direction_2d is not None:
                dx, dy = oblique_direction_2d
                mag = math.hypot(dx, dy)
                if mag > 1e-6:
                    ndx, ndy = dx / mag, dy / mag
                    length = int(min(pix.width(), pix.height()) * 0.6)
                    cxp = pix.width() // 2
                    cyp = pix.height() // 2
                    x1 = int(cxp - ndx * length / 2)
                    y1 = int(cyp - ndy * length / 2)
                    x2 = int(cxp + ndx * length / 2)
                    y2 = int(cyp + ndy * length / 2)
                    pen_o = QPen(QColor(COLORS['Oblique']))
                    pen_o.setWidth(2)
                    painter.setPen(pen_o)
                    painter.drawLine(x1, y1, x2, y2)

            # draw boundary contours if provided
            if boundary_contours:
                pen_b = QPen(QColor('#00FF00'))  # bright green for merged boundary
                pen_b.setWidth(2)
                painter.setPen(pen_b)
                img_w, img_h = w, h
                pix_w, pix_h = pix.width(), pix.height()
                offset_x = (pix_w - img_w * self.zoom) / 2.0
                offset_y = (pix_h - img_h * self.zoom) / 2.0
                for cnt in boundary_contours:
                    if cnt.size == 0:
                        continue
                    prev = None
                    for (r, c) in cnt:
                        x = int(clamp(c * self.zoom + 0.5 + offset_x, 0, pix_w - 1))
                        y = int(clamp(r * self.zoom + 0.5 + offset_y, 0, pix_h - 1))
                        if prev is not None:
                            painter.drawLine(prev[0], prev[1], x, y)
                        prev = (x, y)
                    # close contour (last->first)
                    if len(cnt) >= 2:
                        r0, c0 = cnt[0]
                        x0 = int(clamp(c0 * self.zoom + 0.5 + offset_x, 0, pix_w - 1))
                        y0 = int(clamp(r0 * self.zoom + 0.5 + offset_y, 0, pix_h - 1))
                        painter.drawLine(prev[0], prev[1], x0, y0)

            painter.end()
            self.setPixmap(pix)
        except Exception as e:
            print(f"Error in update_display ({self.view_name}): {e}")
            traceback.print_exc()

    # Mouse handling to drag v/h/oblique
    def mousePressEvent(self, event):
        if self.image_data is None or event.button() != Qt.LeftButton:
            return
        p = self.parent_viewer
        pix = self.pixmap()
        if pix is None:
            return
        lw, lh = self.width(), self.height()
        pw, ph = pix.width(), pix.height()
        offx = (lw - pw) / 2.0
        offy = (lh - ph) / 2.0
        mx = (event.x() - offx) / self.zoom
        my = (event.y() - offy) / self.zoom
        mx = clamp(mx, 0, self.image_data.shape[1] - 1)
        my = clamp(my, 0, self.image_data.shape[0] - 1)

        tol_px = 8
        v_widget_x = int(self.v_pos * self.zoom + offx)
        h_widget_y = int(self.h_pos * self.zoom + offy)
        dx_v = abs(event.x() - v_widget_x)
        dy_h = abs(event.y() - h_widget_y)

        # check oblique proximity
        ob_dir = p.get_oblique_projection_on_view(self.view_name)
        is_near_oblique = False
        if ob_dir is not None:
            obdx, obdy = ob_dir
            mag = math.hypot(obdx, obdy)
            if mag > 1e-6:
                ndx, ndy = obdx / mag, obdy / mag
                cx = int(pw // 2 + offx)
                cy = int(ph // 2 + offy)
                vx = event.x() - cx; vy = event.y() - cy
                perp = abs(-ndy * vx + ndx * vy)
                if perp <= tol_px:
                    is_near_oblique = True

        if is_near_oblique:
            self.dragging = 'oblique'
        else:
            if dx_v <= tol_px and dx_v <= dy_h:
                self.dragging = 'v'
            elif dy_h <= tol_px and dy_h < dx_v:
                self.dragging = 'h'
            else:
                if abs(mx - self.v_pos) < abs(my - self.h_pos):
                    self.dragging = 'v'
                else:
                    self.dragging = 'h'

        p.handle_drag_from_view(self.view_name, self.dragging, mx, my)

    def mouseMoveEvent(self, event):
        if self.image_data is None or self.dragging is None:
            return
        pix = self.pixmap()
        if pix is None:
            return
        lw, lh = self.width(), self.height()
        pw, ph = pix.width(), pix.height()
        offx = (lw - pw) / 2.0
        offy = (lh - ph) / 2.0
        mx = (event.x() - offx) / self.zoom
        my = (event.y() - offy) / self.zoom
        mx = clamp(mx, 0, self.image_data.shape[1] - 1)
        my = clamp(my, 0, self.image_data.shape[0] - 1)
        p = self.parent_viewer
        p.handle_drag_from_view(self.view_name, self.dragging, mx, my)

    def mouseReleaseEvent(self, event):
        self.dragging = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom = min(8.0, self.zoom * 1.15)
        else:
            self.zoom = max(0.2, self.zoom / 1.15)
        self.update_display()


# ---------- Main application ----------
class MPRViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Professional MPR Viewer")
        self.setGeometry(40, 40, 1500, 980)

        # data
        self.volume = None
        self.combined_mask = None
        self.shape = None
        self.last_dicom_ds = None  # keep the last loaded pydicom Dataset for metadata-based AI

        # crosshair in voxel coords (x,y,z)
        self.cross = [0, 0, 0]

        # slice indices mapping: axial z, sagittal x, coronal y
        self.slice_idx = [0, 0, 0]

        # oblique angles
        self.oblique_tilt_x = 0.0
        self.oblique_tilt_y = 30.0

        # playback
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.play_states = {'Axial': False, 'Sagittal': False, 'Coronal': False}

        # UI references
        self.canvas_axial = None
        self.canvas_sagittal = None
        self.canvas_coronal = None
        self.canvas_oblique = None

        self.axial_slider = None; self.sag_slider = None; self.cor_slider = None
        self.speed_slider = None

        self.init_ui()
        self.ensure_default_qss()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        vmain = QVBoxLayout(central)

        # Top controls: load NIfTI, load DICOM, ROI export, speed, oblique tilt
        top_controls = QGroupBox("Controls")
        th = QHBoxLayout(top_controls)

        self.btn_load_nii = QPushButton("Load NIfTI")
        self.btn_load_nii.clicked.connect(self.load_nifti)
        th.addWidget(self.btn_load_nii)

        self.btn_load_dcm = QPushButton("Load DICOM (.dcm)")
        self.btn_load_dcm.clicked.connect(self.load_dicom_file)
        if not DICOM_AVAILABLE:
            self.btn_load_dcm.setEnabled(False)
            self.btn_load_dcm.setToolTip("Install pydicom to enable DICOM support")
        th.addWidget(self.btn_load_dcm)

        # --- ADDED: Predict Orientation button + result label ---
        predict_btn = QPushButton("Predict Orientation")
        predict_btn.clicked.connect(self.on_predict_clicked)
        th.addWidget(predict_btn)
        self.prediction_label = QLabel("Prediction: -")
        self.prediction_label.setStyleSheet("font-weight:bold; padding-left:8px;")
        th.addWidget(self.prediction_label)
        # --- end added ---

        # --- NEW: Detect Organ (DICOM metadata) ---
        self.btn_detect_organ = QPushButton("Detect Organ")
        self.btn_detect_organ.clicked.connect(self.on_detect_organ_clicked)
        th.addWidget(self.btn_detect_organ)
        self.organ_label = QLabel("Organ: -")
        self.organ_label.setStyleSheet("font-weight:bold; padding-left:8px;")
        th.addWidget(self.organ_label)

        th.addStretch()

        th.addWidget(QLabel("Play Speed (FPS):"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1); self.speed_slider.setMaximum(30); self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.update_timer_interval)
        self.speed_slider.setMaximumWidth(140)
        th.addWidget(self.speed_slider)

        th.addWidget(QLabel("Oblique Tilt X:"))
        self.tilt_x_slider = QSlider(Qt.Horizontal)
        self.tilt_x_slider.setMinimum(-89); self.tilt_x_slider.setMaximum(89); self.tilt_x_slider.setValue(int(self.oblique_tilt_x))
        self.tilt_x_slider.valueChanged.connect(self.tilt_x_changed)
        self.tilt_x_slider.setMaximumWidth(140)
        th.addWidget(self.tilt_x_slider)

        th.addWidget(QLabel("Oblique Tilt Y:"))
        self.tilt_y_slider = QSlider(Qt.Horizontal)
        self.tilt_y_slider.setMinimum(-89); self.tilt_y_slider.setMaximum(89); self.tilt_y_slider.setValue(int(self.oblique_tilt_y))
        self.tilt_y_slider.valueChanged.connect(self.tilt_y_changed)
        self.tilt_y_slider.setMaximumWidth(140)
        th.addWidget(self.tilt_y_slider)

        vmain.addWidget(top_controls)

        # Viewport grid
        grid = QGridLayout()
        self.canvas_axial = ImageCanvas('Axial', self)
        self.canvas_sagittal = ImageCanvas('Sagittal', self)
        self.canvas_coronal = ImageCanvas('Coronal', self)
        self.canvas_oblique = ImageCanvas('Oblique', self)

        grid.addWidget(self.wrap_with_controls(self.canvas_axial, 'Axial'), 0, 0)
        grid.addWidget(self.wrap_with_controls(self.canvas_sagittal, 'Sagittal'), 0, 1)
        grid.addWidget(self.wrap_with_controls(self.canvas_coronal, 'Coronal'), 1, 0)
        grid.addWidget(self.wrap_with_controls(self.canvas_oblique, 'Oblique', controls=False), 1, 1)

        vmain.addLayout(grid, stretch=1)

        # ROI export
        roi_group = QGroupBox("ROI Export (axial slices)")
        roi_layout = QHBoxLayout(roi_group)
        roi_layout.addWidget(QLabel("Start slice:"))
        self.roi_start = QSpinBox(); self.roi_start.setMinimum(0)
        roi_layout.addWidget(self.roi_start)
        roi_layout.addWidget(QLabel("End slice:"))
        self.roi_end = QSpinBox(); self.roi_end.setMinimum(0)
        roi_layout.addWidget(self.roi_end)
        self.btn_export = QPushButton("Export ROI")
        self.btn_export.clicked.connect(self.export_roi)
        roi_layout.addWidget(self.btn_export)
        roi_layout.addStretch()
        vmain.addWidget(roi_group)

        self.statusBar().showMessage("Ready")

    def wrap_with_controls(self, canvas, title, controls=True):
        cont = QWidget(); vl = QVBoxLayout(cont); vl.setContentsMargins(0,0,0,0)
        # header
        header = QWidget(); hh = QHBoxLayout(header); hh.setContentsMargins(5,5,5,5)
        sq = QLabel(); sq.setFixedSize(18,12); sq.setStyleSheet(f"background-color: {COLORS.get(title,'#fff')}; border:1px solid #222;")
        lab = QLabel(f"<b>{title} View</b>"); lab.setStyleSheet("color:white;")
        hh.addWidget(sq); hh.addWidget(lab); hh.addStretch()
        # if oblique, add dropdown to select Oblique / Surface Boundary
        if title == 'Oblique':
            lab_mode = QLabel("Mode:")
            self.oblique_mode = QComboBox()
            self.oblique_mode.addItems(["Oblique", "Surface"])
            self.oblique_mode.currentTextChanged.connect(self.on_oblique_mode_changed)
            hh.addWidget(lab_mode); hh.addWidget(self.oblique_mode)
            hh.addStretch()
        vl.addWidget(header)
        vl.addWidget(canvas)

        if controls:
            ctrl = QWidget(); ch = QHBoxLayout(ctrl); ch.setContentsMargins(4,4,4,4)
            back = QPushButton("◀"); play = QPushButton("Play"); play.setCheckable(True); fwd = QPushButton("▶")
            slider = QSlider(Qt.Horizontal)
            ch.addWidget(back); ch.addWidget(play); ch.addWidget(fwd); ch.addWidget(QLabel("Slice:")); ch.addWidget(slider)
            vl.addWidget(ctrl)

            if title == 'Axial':
                back.clicked.connect(lambda: self.step_view('Axial', -1))
                fwd.clicked.connect(lambda: self.step_view('Axial', 1))
                play.clicked.connect(lambda checked: self.toggle_play('Axial', checked, play))
                slider.valueChanged.connect(lambda v: self.slider_moved('Axial', v))
                self.axial_slider = slider; self.axial_play_btn = play
            elif title == 'Sagittal':
                back.clicked.connect(lambda: self.step_view('Sagittal', -1))
                fwd.clicked.connect(lambda: self.step_view('Sagittal', 1))
                play.clicked.connect(lambda checked: self.toggle_play('Sagittal', checked, play))
                slider.valueChanged.connect(lambda v: self.slider_moved('Sagittal', v))
                self.sag_slider = slider; self.sag_play_btn = play
            elif title == 'Coronal':
                back.clicked.connect(lambda: self.step_view('Coronal', -1))
                fwd.clicked.connect(lambda: self.step_view('Coronal', 1))
                play.clicked.connect(lambda checked: self.toggle_play('Coronal', checked, play))
                slider.valueChanged.connect(lambda v: self.slider_moved('Coronal', v))
                self.cor_slider = slider; self.cor_play_btn = play

        return cont

    # ================= Loading files =================
    def load_nifti(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if not path:
            return
        self.statusBar().showMessage(f"Loading {os.path.basename(path)}...")
        QApplication.processEvents()
        try:
            nii = nib.load(path)
            data = np.asarray(nii.dataobj)
            if data.ndim == 4:
                data = data[..., 0]
                QMessageBox.information(self, "Info", "4D volume: using first timepoint")
            if data.ndim == 2:
                data = data[..., np.newaxis]
            if data.ndim != 3:
                QMessageBox.critical(self, "Error", f"Unsupported shape: {data.shape}")
                return
            self.volume = data.astype(np.float32)
            nx, ny, nz = self.volume.shape
            self.shape = (nx, ny, nz)
            self.cross = [nx//2, ny//2, nz//2]
            self.slice_idx = [nz//2, nx//2, ny//2]
            # update sliders if present
            if self.axial_slider:
                self.axial_slider.blockSignals(True); self.axial_slider.setMinimum(0); self.axial_slider.setMaximum(nz-1)
                self.axial_slider.setValue(self.slice_idx[0]); self.axial_slider.blockSignals(False)
            if self.sag_slider:
                self.sag_slider.blockSignals(True); self.sag_slider.setMinimum(0); self.sag_slider.setMaximum(nx-1)
                self.sag_slider.setValue(self.slice_idx[1]); self.sag_slider.blockSignals(False)
            if self.cor_slider:
                self.cor_slider.blockSignals(True); self.cor_slider.setMinimum(0); self.cor_slider.setMaximum(ny-1)
                self.cor_slider.setValue(self.slice_idx[2]); self.cor_slider.blockSignals(False)
            # setup ROI spinboxes
            self.roi_start.setMaximum(nz-1); self.roi_end.setMaximum(nz-1); self.roi_end.setValue(nz-1)
            self.update_all_views()
            self.statusBar().showMessage(f"Loaded volume shape: {self.volume.shape}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load NIfTI: {e}")
            print(e)
        finally:
            self.statusBar().showMessage("Ready")

    def load_dicom_file(self):
        if not DICOM_AVAILABLE:
            QMessageBox.warning(self, "Warning", "pydicom not available")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open DICOM File", "", "DICOM Files (*.dcm)")
        if not path:
            return
        try:
            ds = pydicom.dcmread(path)
            self.last_dicom_ds = ds
            arr = ds.pixel_array.astype(np.float32)
            # if single 2D image, wrap into z dimension = 1
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            if arr.ndim == 3:
                self.volume = arr.astype(np.float32)
                nx, ny, nz = self.volume.shape
                self.shape = (nx, ny, nz)
                self.cross = [nx//2, ny//2, nz//2]
                self.slice_idx = [nz//2, nx//2, ny//2]
                self.update_all_views()
                self.statusBar().showMessage("Loaded DICOM pixel_array (single file)")
            else:
                QMessageBox.information(self, "Info", "Loaded DICOM but shape not 2D/3D")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read DICOM: {e}")
            print(e)

    # ================= Oblique plane math =================
    def get_oblique_plane_normal(self):
        tx = math.radians(self.oblique_tilt_x)
        ty = math.radians(self.oblique_tilt_y)
        Rx = np.array([[1,0,0],[0, math.cos(tx), -math.sin(tx)],[0, math.sin(tx), math.cos(tx)]], dtype=np.float32)
        Ry = np.array([[math.cos(ty), 0, math.sin(ty)],[0,1,0],[-math.sin(ty),0, math.cos(ty)]], dtype=np.float32)
        n0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        n = Ry.dot(Rx.dot(n0))
        return n

    def get_oblique_projection_on_view(self, view_name):
        n = self.get_oblique_plane_normal()
        if self.volume is None:
            return None
        if view_name == 'Axial':
            view_normal = np.array([0.0,0.0,1.0])
            d = np.cross(n, view_normal)
            return (float(d[0]), float(-d[1]))
        elif view_name == 'Sagittal':
            view_normal = np.array([1.0,0.0,0.0])
            d = np.cross(n, view_normal)
            return (float(d[1]), float(d[2]))
        elif view_name == 'Coronal':
            view_normal = np.array([0.0,1.0,0.0])
            d = np.cross(n, view_normal)
            return (float(d[0]), float(d[2]))
        else:
            return None

    def sample_oblique(self, center, width=None, height=None):
        vol = self.volume
        nx, ny, nz = vol.shape
        if width is None: width = ny
        if height is None: height = nx
        n = self.get_oblique_plane_normal()
        up = np.array([0.0,1.0,0.0])
        if abs(np.dot(n, up)) > 0.9:
            up = np.array([1.0,0.0,0.0])
        e1 = np.cross(up, n); e1 = e1 / (np.linalg.norm(e1) + 1e-12)
        e2 = np.cross(n, e1); e2 = e2 / (np.linalg.norm(e2) + 1e-12)
        i_coords = (np.arange(height) - (height - 1) / 2.0)
        j_coords = (np.arange(width) - (width - 1) / 2.0)
        ii, jj = np.meshgrid(i_coords, j_coords, indexing='ij')
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        X = cx + ii * e1[0] + jj * e2[0]
        Y = cy + ii * e1[1] + jj * e2[1]
        Z = cz + ii * e1[2] + jj * e2[2]
        coords = np.vstack((X.ravel(), Y.ravel(), Z.ravel()))
        sampled = map_coordinates(vol, coords, order=1, mode='nearest')
        sampled = sampled.reshape((height, width))
        return sampled

    # ================= Drag handlers (synchronization) =================
    def handle_drag_from_view(self, view_name, dragging_type, img_x, img_y):
        if self.volume is None:
            return
        nx, ny, nz = self.shape
        if view_name == 'Axial':
            if dragging_type == 'v':
                self.cross[0] = int(clamp(round(img_x), 0, nx - 1))
            elif dragging_type == 'h':
                self.cross[1] = int(clamp(round(img_y), 0, ny - 1))
            elif dragging_type == 'oblique':
                w = self.canvas_axial.image_data.shape[1]; h = self.canvas_axial.image_data.shape[0]
                cx = w / 2.0; cy = h / 2.0
                ang = math.degrees(math.atan2(img_y - cy, img_x - cx))
                self.oblique_tilt_y = clamp(ang, -89, 89)
                self.tilt_y_slider.blockSignals(True); self.tilt_y_slider.setValue(int(self.oblique_tilt_y)); self.tilt_y_slider.blockSignals(False)
        elif view_name == 'Sagittal':
           if dragging_type == 'v':
            self.cross[1] = int(clamp(round(img_x), 0, ny - 1))
           elif dragging_type == 'h':
            # Flip the coordinate back
              self.cross[2] = int(clamp(nz - 1 - round(img_y), 0, nz - 1))
              # ... rest of code
           elif dragging_type == 'oblique':
                w = self.canvas_sagittal.image_data.shape[1]; h = self.canvas_sagittal.image_data.shape[0]
                cx = w / 2.0; cy = h / 2.0
                ang = math.degrees(math.atan2(img_y - cy, img_x - cx))
                self.oblique_tilt_x = clamp(-ang, -89, 89)
                self.tilt_x_slider.blockSignals(True); self.tilt_x_slider.setValue(int(self.oblique_tilt_x)); self.tilt_x_slider.blockSignals(False)
        elif view_name == 'Coronal':
           if dragging_type == 'v':
             self.cross[0] = int(clamp(round(img_x), 0, nx - 1))
           elif dragging_type == 'h':
                # Flip the coordinate back
              self.cross[2] = int(clamp(nz - 1 - round(img_y), 0, nz - 1))
                # ... rest of code
           elif dragging_type == 'oblique':
                w = self.canvas_coronal.image_data.shape[1]; h = self.canvas_coronal.image_data.shape[0]
                cx = w / 2.0; cy = h / 2.0
                ang = math.degrees(math.atan2(img_y - cy, img_x - cx))
                self.oblique_tilt_x = clamp(ang, -89, 89)
                self.tilt_x_slider.blockSignals(True); self.tilt_x_slider.setValue(int(self.oblique_tilt_x)); self.tilt_x_slider.blockSignals(False)

        # synchronize slice indices
        self.slice_idx[0] = int(clamp(self.cross[2], 0, nz - 1))
        self.slice_idx[1] = int(clamp(self.cross[0], 0, nx - 1))
        self.slice_idx[2] = int(clamp(self.cross[1], 0, ny - 1))
        self.update_all_views()

    # ================= Slider / play handlers =================
    def slider_moved(self, view_name, val):
        if self.volume is None: return
        nx, ny, nz = self.shape
        if view_name == 'Axial':
            self.slice_idx[0] = clamp(val, 0, nz-1); self.cross[2] = int(self.slice_idx[0])
        elif view_name == 'Sagittal':
            self.slice_idx[1] = clamp(val, 0, nx-1); self.cross[0] = int(self.slice_idx[1])
        elif view_name == 'Coronal':
            self.slice_idx[2] = clamp(val, 0, ny-1); self.cross[1] = int(self.slice_idx[2])
        self.update_all_views()

    def toggle_play(self, view_name, checked, btn):
        if view_name == 'Axial':
            self.play_states['Axial'] = checked; btn.setText('Pause' if checked else 'Play')
        elif view_name == 'Sagittal':
            self.play_states['Sagittal'] = checked; btn.setText('Pause' if checked else 'Play')
        elif view_name == 'Coronal':
            self.play_states['Coronal'] = checked; btn.setText('Pause' if checked else 'Play')
        self.refresh_timer()

    def refresh_timer(self):
        any_play = any(self.play_states.values())
        if any_play and not self.timer.isActive():
            self.update_timer_interval(); self.timer.start()
        elif not any_play and self.timer.isActive():
            self.timer.stop()

    def update_timer_interval(self):
        fps = max(1, self.speed_slider.value())
        self.timer.setInterval(int(1000.0 / fps))

    def on_timer(self):
        if self.play_states.get('Axial'):
            self.step_view('Axial', 1)
        if self.play_states.get('Sagittal'):
            self.step_view('Sagittal', 1)
        if self.play_states.get('Coronal'):
            self.step_view('Coronal', 1)

    def step_view(self, view_name, step):
        if self.volume is None: return
        nx, ny, nz = self.shape
        if view_name == 'Axial':
            self.slice_idx[0] = int(clamp(self.slice_idx[0] + step, 0, nz - 1)); self.cross[2] = self.slice_idx[0]
        elif view_name == 'Sagittal':
            self.slice_idx[1] = int(clamp(self.slice_idx[1] + step, 0, nx - 1)); self.cross[0] = self.slice_idx[1]
        elif view_name == 'Coronal':
            self.slice_idx[2] = int(clamp(self.slice_idx[2] + step, 0, ny - 1)); self.cross[1] = self.slice_idx[2]
        self.update_all_views()

    def tilt_x_changed(self, v):
        self.oblique_tilt_x = float(v); self.update_all_views()
    def tilt_y_changed(self, v):
        self.oblique_tilt_y = float(v); self.update_all_views()

    # ================= Update / render views =================
    def update_all_views(self):
        if self.volume is None:
            return
        nx, ny, nz = self.shape
        self.cross[0] = int(clamp(self.cross[0], 0, nx-1))
        self.cross[1] = int(clamp(self.cross[1], 0, ny-1))
        self.cross[2] = int(clamp(self.cross[2], 0, nz-1))
        self.slice_idx[0] = int(clamp(self.slice_idx[0], 0, nz-1))
        self.slice_idx[1] = int(clamp(self.slice_idx[1], 0, nx-1))
        self.slice_idx[2] = int(clamp(self.slice_idx[2], 0, ny-1))

        axial = self.volume[:, :, self.slice_idx[0]]
        sagittal = np.fliplr(self.volume[self.slice_idx[1], :, :])
        coronal = np.fliplr(self.volume[:, self.slice_idx[2], :])

        ob = self.sample_oblique(center=self.cross, width=coronal.shape[1], height=coronal.shape[0])

        self.canvas_axial.set_image(axial.T, v_pos=self.cross[0], h_pos=self.cross[1])
           # Flip the h_pos for sagittal and coronal to match the flipped image
        self.canvas_sagittal.set_image(sagittal.T, v_pos=self.cross[1], h_pos=nz - 1 - self.cross[2])
        self.canvas_coronal.set_image(coronal.T, v_pos=self.cross[0], h_pos=nz - 1 - self.cross[2])
        if getattr(self, "oblique_mode", None) and self.oblique_mode.currentText() == "Surface":
            black = np.zeros_like(ob.T, dtype=np.float32)
            self.canvas_oblique.set_image(black, v_pos=ob.shape[1]//2, h_pos=ob.shape[0]//2)
        else:
            self.canvas_oblique.set_image(ob.T, v_pos=ob.shape[1]//2, h_pos=ob.shape[0]//2)

        proj_ax = self.get_oblique_projection_on_view('Axial')
        proj_sg = self.get_oblique_projection_on_view('Sagittal')
        proj_cr = self.get_oblique_projection_on_view('Coronal')

        boundary_contours = None
        if getattr(self, 'oblique_mode', None) and self.oblique_mode.currentText() == "Surface" and self.combined_mask is not None:
            z = int(self.slice_idx[0])
            if z < self.combined_mask.shape[2]:
                mask_slice = self.combined_mask[:, :, z]
                contours = measure.find_contours(mask_slice.astype(np.uint8), 0.5)
                boundary_contours = contours
                self.canvas_oblique.update_display(oblique_direction_2d=None, boundary_contours=boundary_contours)
            else:
                self.canvas_axial.update_display(oblique_direction_2d=proj_ax)
                self.canvas_sagittal.update_display(oblique_direction_2d=proj_sg)
                self.canvas_coronal.update_display(oblique_direction_2d=proj_cr)
                self.canvas_oblique.update_display(oblique_direction_2d=None)
        else:
            self.canvas_axial.update_display(oblique_direction_2d=proj_ax)
            self.canvas_sagittal.update_display(oblique_direction_2d=proj_sg)
            self.canvas_coronal.update_display(oblique_direction_2d=proj_cr)
            self.canvas_oblique.update_display(oblique_direction_2d=None)

        if self.axial_slider:
            self.axial_slider.blockSignals(True); self.axial_slider.setValue(self.slice_idx[0]); self.axial_slider.blockSignals(False)
        if self.sag_slider:
            self.sag_slider.blockSignals(True); self.sag_slider.setValue(self.slice_idx[1]); self.sag_slider.blockSignals(False)
        if self.cor_slider:
            self.cor_slider.blockSignals(True); self.cor_slider.setValue(self.slice_idx[2]); self.cor_slider.blockSignals(False)

    def overlay(self, img, seg):
        if seg is None: return img
        if np.any(seg > 0):
            res = img.astype(np.float32).copy()
            mask = seg > 0
            res[mask] = np.maximum(res[mask], seg[mask] * 100.0)
            return res
        return img

    # ================= Oblique mode dropdown =================
    def on_oblique_mode_changed(self, text):
        if text == "Surface":
            files, _ = QFileDialog.getOpenFileNames(self, "Select NIfTI mask files (1 or more)", "", "NIfTI Files (*.nii *.nii.gz)")
            if not files:
                self.oblique_mode.blockSignals(True)
                self.oblique_mode.setCurrentText("Oblique")
                self.oblique_mode.blockSignals(False)
                return
            combined = None
            for f in files:
                try:
                    nii = nib.load(f)
                    m = np.asarray(nii.dataobj)
                    if m.ndim == 2:
                        m = m[:, :, np.newaxis]
                    if m.ndim != 3:
                        QMessageBox.warning(self, "Warning", f"Skipping file (unsupported shape): {os.path.basename(f)}")
                        continue
                    if combined is None:
                        combined = (m != 0)
                    else:
                        if combined.shape != m.shape:
                            QMessageBox.warning(self, "Warning", f"Mask {os.path.basename(f)} shape {m.shape} mismatches previous {combined.shape}; skipping")
                            continue
                        combined = np.logical_or(combined, (m != 0))
                except Exception as e:
                    QMessageBox.warning(self, "Warning", f"Failed to load mask {os.path.basename(f)}: {e}")
            if combined is None:
                QMessageBox.information(self, "Info", "No valid masks loaded; reverting to Oblique View")
                self.oblique_mode.blockSignals(True)
                self.oblique_mode.setCurrentText("Oblique")
                self.oblique_mode.blockSignals(False)
                return
            self.combined_mask = combined.astype(np.uint8)
            if self.volume is None:
                self.volume = np.zeros_like(self.combined_mask, dtype=np.float32)
                self.shape = self.volume.shape
                self.cross = [self.shape[0]//2, self.shape[1]//2, self.shape[2]//2]
                self.slice_idx = [self.shape[2]//2, self.shape[0]//2, self.shape[1]//2]
            else:
                if self.combined_mask.shape != self.volume.shape:
                    QMessageBox.warning(self, "Warning", "Combined mask shape does not match current volume. Mask will still be used for boundary display (slice index mapping)")
            self.update_all_views()
        else:
            self.update_all_views()

    # ================= ROI export =================
    def export_roi(self):
        if self.volume is None:
            QMessageBox.warning(self, "Warning", "No volume loaded")
            return
        start = self.roi_start.value(); end = self.roi_end.value()
        if start >= end:
            QMessageBox.warning(self, "Warning", "Start must be < End")
            return
        if end > self.volume.shape[2] - 1:
            QMessageBox.warning(self, "Warning", f"End exceeds depth ({self.volume.shape[2]})")
            return
        roi = self.volume[:, :, start:end+1]
        path, _ = QFileDialog.getSaveFileName(self, "Save ROI", "", "NIfTI Files (*.nii.gz *.nii)")
        if not path:
            return
        try:
            nib.save(nib.Nifti1Image(roi, np.eye(4)), path)
            QMessageBox.information(self, "Saved", f"Saved ROI to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save ROI: {e}")


    # ================= AI ORGAN PREDICTION FROM DICOM METADATA =================
    def _get_ds_str(self, ds, name, default=""):
        try:
            v = getattr(ds, name, None)
            if v is None:
                return default
            sv = str(v).strip()
            return sv if sv else default
        except Exception:
            return default

    def predict_organ_from_metadata(self, ds):
        """
        Heuristic organ detector using common DICOM tags:
        BodyPartExamined, StudyDescription, SeriesDescription, ProtocolName, ViewPosition.
        Returns (organ_label, details_dict).
        """
        body = self._get_ds_str(ds, "BodyPartExamined").lower()
        prot = self._get_ds_str(ds, "ProtocolName").lower()
        ser  = self._get_ds_str(ds, "SeriesDescription").lower()
        stdy = self._get_ds_str(ds, "StudyDescription").lower()
        view = self._get_ds_str(ds, "ViewPosition").lower()
        blob = " ".join([body, prot, ser, stdy, view])

        # Priority 1: BodyPartExamined exact-ish
        mapping = {
            "head": "Head/Brain", "brain": "Head/Brain", "skull": "Head/Brain",
            "cspine": "Cervical Spine", "tspine": "Thoracic Spine", "lspine": "Lumbar Spine", "spine": "Spine",
            "chest": "Chest/Lung", "thorax": "Chest/Lung", "lung": "Chest/Lung",
            "abdomen": "Abdomen", "pelvis": "Pelvis", "abdo-pel": "Abdomen/Pelvis", "abdominopelvic": "Abdomen/Pelvis",
            "hip": "Hip", "knee": "Knee", "foot": "Foot", "ankle": "Ankle",
            "shoulder": "Shoulder", "elbow": "Elbow", "wrist": "Wrist", "hand": "Hand",
            "neck": "Neck", "sinus": "Sinus", "temporal bone": "Temporal Bone", "orbits": "Orbits",
            "heart": "Heart", "liver": "Liver", "kidney": "Kidney", "pancreas": "Pancreas",
            "prostate": "Prostate", "breast": "Breast", "cta chest": "Chest/Lung", "cta head": "Head/Brain",
            "cta neck": "Neck"
        }
        if body:
            for k, v in mapping.items():
                if k in body:
                    return v, {"source": "BodyPartExamined", "value": body}

        # Priority 2: keywords in other descriptions
        keywords = [
            ("brain", "Head/Brain"), ("head", "Head/Brain"),
            ("c-spine", "Cervical Spine"), ("t-spine", "Thoracic Spine"), ("l-spine", "Lumbar Spine"), ("spine", "Spine"),
            ("chest", "Chest/Lung"), ("thorax", "Chest/Lung"), ("lung", "Chest/Lung"), ("pe protocol", "Chest/Lung"), ("cta chest", "Chest/Lung"),
            ("abdomen", "Abdomen"), ("liver", "Liver"), ("kidney", "Kidney"), ("pancreas", "Pancreas"), ("spleen", "Spleen"),
            ("pelvis", "Pelvis"), ("urogram", "Urinary Tract"), ("a/p", "Abdomen/Pelvis"), ("abdopel", "Abdomen/Pelvis"),
            ("hip", "Hip"), ("knee", "Knee"), ("ankle", "Ankle"), ("foot", "Foot"),
            ("shoulder", "Shoulder"), ("elbow", "Elbow"), ("wrist", "Wrist"), ("hand", "Hand"),
            ("neck", "Neck"), ("sinus", "Sinus"), ("orbit", "Orbits"), ("temporal", "Temporal Bone"),
            ("aorta", "Aorta"), ("coronary", "Coronary"), ("heart", "Heart"),
            ("prostate", "Prostate"), ("breast", "Breast")
        ]
        for kw, label in keywords:
            if kw in blob:
                return label, {"source": "Series/Protocol/Study", "match": kw, "blob": blob}

        # ViewPosition hint (CXR)
        if view in ("pa", "ap", "ap supine", "lateral") and ("chest" in blob or "lung" in blob):
            return "Chest/Lung", {"source": "ViewPosition", "value": view}

        return "Unknown", {"source": "none", "blob": blob}

    def on_detect_organ_clicked(self):
        if not DICOM_AVAILABLE:
            QMessageBox.warning(self, "Warning", "pydicom not available")
            return
        ds = self.last_dicom_ds
        if ds is None:
            path, _ = QFileDialog.getOpenFileName(self, "Open DICOM for metadata", "", "DICOM Files (*.dcm)")
            if not path:
                return
            try:
                ds = pydicom.dcmread(path)
                self.last_dicom_ds = ds
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read DICOM: {e}")
                return
        organ, details = self.predict_organ_from_metadata(ds)
        msg = f"Organ: {organ}"
        self.organ_label.setText(f"Organ: {organ}")
        QMessageBox.information(self, "Organ", msg)

    # ================= AI ORIENTATION PREDICTION =================
    def on_predict_clicked(self):
        """Run orientation prediction on the currently loaded 3D volume."""
        if self.volume is None:
            QMessageBox.warning(self, "Warning", "Load a NIfTI volume first.")
            return

        global orientation_model
        if orientation_model is None:
            orientation_model = load_orientation_model(".")
            if orientation_model is None:
                QMessageBox.critical(self, "Error", "Orientation model not available (failed to load).")
                return

        try:
            inp = getattr(orientation_model, "input_shape", None)
            if inp is None and hasattr(orientation_model, "_inferred_input_shape"):
                inp = orientation_model._inferred_input_shape
            if inp is None:
                inp = (None, 128, 128, 128, 1)

            if isinstance(inp, (list, tuple)) and len(inp) >= 2:
                shape = inp[1:]
            else:
                shape = tuple(inp)

            # detect target dims (H,W,D,C) or (C,H,W,D)
            channels_first = False
            c = 1
            if len(shape) == 4:
                if shape[0] in (1, 3):
                    channels_first = True
                    c = int(shape[0]); th = int(shape[1]); tw = int(shape[2]); td = int(shape[3])
                else:
                    channels_first = False
                    th = int(shape[0]); tw = int(shape[1]); td = int(shape[2]); c = int(shape[3])
            elif len(shape) == 3:
                th, tw, td = int(shape[0]), int(shape[1]), int(shape[2]); c = 1
            else:
                th, tw, td, c = 128, 128, 128, 1

            target = (th, tw, td)
            vol = self.volume.astype(np.float32)
            resized = resize_volume_to_target(vol, target)

            if c == 1:
                x = resized[np.newaxis, ..., np.newaxis]
            else:
                x = np.stack([resized] * c, axis=-1)[np.newaxis, ...]
            if channels_first:
                x = np.transpose(x, (0, 4, 1, 2, 3))

            preds = orientation_model.predict(x)
            if preds.ndim == 2 and preds.shape[1] >= 3:
                probs = np.asarray(preds[0], dtype=np.float32)
                if probs.min() < 0 or probs.max() > 1.001:
                    ex = np.exp(probs - np.max(probs))
                    probs = ex / np.sum(ex)
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
            else:
                idx = int(np.argmax(preds))
                conf = float(np.max(preds))

            labels = ["Axial", "Coronal", "Sagittal"]
            label = labels[idx] if idx < len(labels) else f"class_{idx}"
            text = f"Prediction: {label}  (confidence: {conf:.2f})"
            try:
                self.prediction_label.setText(text)
            except Exception:
                pass
            QMessageBox.information(self, "Prediction", text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {e}")
            print("Prediction error:", e)

    # ================= Helpers =================
    def update_window_level(self, *args):
        self.update_all_views()

    def ensure_default_qss(self):
        qss_name = "dark_red.qss"
        if not os.path.exists(qss_name):
            default_qss = """
/* A compact dark-red theme for the viewer */
QMainWindow, QWidget { background-color: #1b1b1b; color: #eaeaea; }
QGroupBox { border: 1px solid #3a3a3a; border-radius: 6px; margin-top: 8px; padding: 6px; background:#222; }
QPushButton { background:#7a0f12; color:white; border-radius:6px; padding:6px 10px; }
QPushButton:hover { background:#a8181d; }
QLabel { color:#eaeaea; }
QSlider::handle:horizontal { background:#cc0000; border: 1px solid #fff;}
"""
            try:
                with open(qss_name, "w", encoding="utf-8") as f:
                    f.write(default_qss)
                print(f"Wrote default theme to {qss_name}")
            except Exception as e:
                print("Failed to write default qss:", e)


# ---------- main ----------
def main():
    app = QApplication(sys.argv)
    qss_file = "dark_red.qss"
    if os.path.exists(qss_file):
        try:
            with open(qss_file, "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())
        except Exception:
            pass
    w = MPRViewer()
    w.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()