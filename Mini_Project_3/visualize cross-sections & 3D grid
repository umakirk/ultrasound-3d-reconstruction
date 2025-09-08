import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QSlider, QLabel, QPushButton, QSizePolicy, QComboBox, QLineEdit
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from vispy import scene
from vispy.app import use_app
from vispy.scene.cameras import TurntableCamera
from vispy.visuals.transforms import STTransform


class MultiPlaneViewer(QWidget):
    def __init__(self, volume):
        super().__init__()
        self.volume = volume
        self.shape = volume.shape
        self.indices = [
            self.shape[2] // 2,  # Z index for XY
            self.shape[1] // 2,  # Y index for XZ
            self.shape[0] // 2   # X index for YZ
        ]
        self.focus = 0
        self.expanded_view = None

        self.labels = ['Z (XY)', 'Y (XZ)', 'X (YZ)']
        self.layout = QVBoxLayout(self)
        self.containers = []
        self.canvases = []
        self.axes = []
        self.sliders = []
        self.expand_buttons = []

        for i, label in enumerate(self.labels):
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0, 0, 0, 0)

            fig = Figure(figsize=(6, 4.5))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            canvas.setFocusPolicy(Qt.ClickFocus)
            canvas.setFocus()
            canvas.mpl_connect("button_press_event", lambda event, idx=i: self._on_click(event, idx))
            canvas.mpl_connect("scroll_event", lambda event, idx=i: self._on_scroll(event, idx))
            vbox.addWidget(canvas)

            slider_label = QLabel(label)
            slider = QSlider(Qt.Horizontal)
            slider.setMaximum(self.shape[2 - i] - 1)
            slider.setValue(self.indices[i])
            slider.valueChanged.connect(lambda val, axis=i: self.set_index(axis, val))

            btn = QPushButton("Expand")
            btn.clicked.connect(lambda checked, idx=i: self.toggle_expand(idx))

            controls_layout = QHBoxLayout()
            controls_layout.addStretch()
            controls_layout.addWidget(slider)
            controls_layout.addSpacing(20)
            controls_layout.addWidget(btn)
            controls_layout.addStretch()

            slider.setMaximumWidth(300)
            slider.setMinimumWidth(200)
            btn.setMaximumWidth(100)

            vbox.addLayout(controls_layout)
            self.containers.append(container)
            self.canvases.append(canvas)
            self.axes.append(ax)
            self.sliders.append(slider)
            self.expand_buttons.append(btn)

            self.layout.addWidget(container)

        self.update_all()

    def _on_click(self, event, idx):
        if event.inaxes in [self.axes[idx]]:
            self.focus = idx

    def _on_scroll(self, event, idx):
        if self.focus != idx:
            return
        direction = 1 if event.button == 'up' else -1
        max_index = self.sliders[idx].maximum()
        new_val = np.clip(self.indices[idx] + direction, 0, max_index)
        self.sliders[idx].setValue(new_val)

    def keyPressEvent(self, event):
        if self.focus is None:
            return super().keyPressEvent(event)
        key = event.key()
        if key in (Qt.Key_Left, Qt.Key_Down):
            new_val = max(0, self.indices[self.focus] - 1)
        elif key in (Qt.Key_Right, Qt.Key_Up):
            new_val = min(self.sliders[self.focus].maximum(), self.indices[self.focus] + 1)
        else:
            return super().keyPressEvent(event)
        self.sliders[self.focus].setValue(new_val)

    def toggle_expand(self, view_index):
        self.expanded_view = None if self.expanded_view == view_index else view_index
        self.redraw_layout()

    def redraw_layout(self):
        for i, container in enumerate(self.containers):
            visible = self.expanded_view is None or self.expanded_view == i
            container.setVisible(visible)
            self.expand_buttons[i].setText("Return" if self.expanded_view == i else "Expand")

    def update_all(self):
        z, y, x = self.indices
        self.axes[0].clear()
        self.axes[0].imshow(self.volume[:, :, z], cmap='gray', origin='lower')
        self.axes[0].set_title(f"XY slice (Z={z})", color='red')
        self.canvases[0].draw()

        self.axes[1].clear()
        self.axes[1].imshow(self.volume[:, y, :], cmap='gray', origin='lower')
        self.axes[1].set_title(f"XZ slice (Y={y})", color='green')
        self.canvases[1].draw()

        self.axes[2].clear()
        self.axes[2].imshow(self.volume[x, :, :], cmap='gray', origin='lower')
        self.axes[2].set_title(f"YZ slice (X={x})", color='blue')
        self.canvases[2].draw()

    def set_index(self, axis, val):
        self.indices[axis] = val
        self.update_all()


class MainWindow(QMainWindow):
    def __init__(self, volume):
        super().__init__()
        self.setWindowTitle("2D Viewer (Left) + 3D VisPy (Right)")

        central = QWidget()
        self.setCentralWidget(central)
        hbox = QHBoxLayout(central)

        self.mpl_viewer = MultiPlaneViewer(volume)
        hbox.addWidget(self.mpl_viewer, stretch=1)

        right_container = QWidget()
        right_layout = QHBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(right_container, stretch=1)

        self.canvas_3d = scene.SceneCanvas(keys='interactive', show=False, bgcolor='black')
        self.canvas_3d.create_native()
        self.canvas_3d.native.setMinimumSize(400, 400)
        self.canvas_3d.native.setParent(self)
        right_layout.addWidget(self.canvas_3d.native, stretch=1)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        controls_layout.setAlignment(Qt.AlignTop)
        right_layout.addWidget(controls_widget)

        controls_layout.addWidget(QLabel("Rendering Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(['mip', 'translucent', 'iso', 'additive'])
        controls_layout.addWidget(self.method_combo)

        controls_layout.addWidget(QLabel("Iso Threshold:"))
        self.threshold_input = QLineEdit("0.01")
        self.threshold_input.setEnabled(False)
        controls_layout.addWidget(self.threshold_input)

        legend_label = QLabel()
        legend_label.setText(
            '<span style="color:black"> Axes Colors: </span>'
            '<span style="color:blue">X </span>'
            '<span style="color:green">Y </span>'
            '<span style="color:red">Z </span>'
        )
        legend_label.setTextFormat(Qt.RichText)
        controls_layout.addWidget(legend_label)

        reset_button = QPushButton("Reset View")
        reset_button.clicked.connect(self.reset_camera_view)
        controls_layout.addWidget(reset_button)

        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        self.threshold_input.editingFinished.connect(self.on_threshold_changed)

        self.view = self.canvas_3d.central_widget.add_view()
        self.view.camera = TurntableCamera(fov=60, elevation=30, azimuth=45, up='+z')
        self.original_camera_settings = {'elevation': 30, 'azimuth': 45, 'roll': 0, 'fov': 60}

        self.volume_visual = scene.visuals.Volume(
            volume, parent=self.view.scene, threshold=0.01, method='mip')
        self.view.camera.set_range()

        scale = min(volume.shape) // 3
        self.axes = scene.visuals.XYZAxis(parent=self.view.scene)
        self.axes.transform = STTransform(scale=(scale, scale, scale), translate=(0, 0, 0))

    def on_method_changed(self, method):
        self.threshold_input.setEnabled(method == 'iso')
        try:
            threshold = float(self.threshold_input.text())
        except ValueError:
            threshold = 0.01
            self.threshold_input.setText(str(threshold))
        self.volume_visual.method = method
        self.volume_visual.threshold = threshold
        self.canvas_3d.update()

    def on_threshold_changed(self):
        if self.method_combo.currentText() == 'iso':
            try:
                threshold = float(self.threshold_input.text())
            except ValueError:
                threshold = 0.01
                self.threshold_input.setText(str(threshold))
            self.volume_visual.threshold = threshold
            self.canvas_3d.update()

    def reset_camera_view(self):
        for k, v in self.original_camera_settings.items():
            setattr(self.view.camera, k, v)
        self.view.camera.set_range()
        self.canvas_3d.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    volume = np.load("test_files/kidney.npy")
    window = MainWindow(volume)


    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())
