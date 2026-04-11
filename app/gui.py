import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from common import SPEED_OF_LIGHT, setup_logging, load_array_from_csv
from util import calculate_delta_x
from calc_1d import compute_pattern

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background calculation thread
# ---------------------------------------------------------------------------


class CalcWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            p = self.params

            if p.get("csv_path"):
                x_arr, amplitudes = load_array_from_csv(Path(p["csv_path"]))
                logger.info("CSV mode: %s, N=%d", p["csv_path"], len(x_arr))
            else:
                wave_length = SPEED_OF_LIGHT / p["freq"]
                steer_x = np.radians(p["steer_x"])
                delta_x = calculate_delta_x(wave_length, steer_x)
                N = p["N"]
                L = delta_x * (N - 1)
                x_arr = np.array([i * delta_x - L / 2 for i in range(N)])
                amplitudes = np.ones(N)
                logger.info("Uniform mode: N=%d, freq=%.2e", N, p["freq"])

            result = compute_pattern(
                x_arr,
                amplitudes,
                p["freq"],
                p["n_points"],
                p["elem_pattern"],
            )
            self.finished.emit(dict(result))
        except Exception as e:
            logger.error("Calculation error: %s", e)
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Plot canvas helpers
# ---------------------------------------------------------------------------


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, projection=None):
        self.fig = Figure(tight_layout=True)
        if projection:
            self.ax = self.fig.add_subplot(111, projection=projection)
        else:
            self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def clear(self):
        self.ax.cla()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._csv_path = None
        self.setWindowTitle("Antenna Array Calculator")
        self.resize(1200, 700)
        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Controls ---
        controls = QWidget()
        controls.setFixedWidth(260)
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ── Режим ────────────────────────────────────────────────────────
        group_mode = QGroupBox("Режим")
        mode_layout = QVBoxLayout(group_mode)
        self.chk_csv = QCheckBox("Из CSV (произвольное расположение)")
        self.chk_csv.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.chk_csv)

        # ── Равномерная решётка ───────────────────────────────────────────
        self.group_uniform = QGroupBox("Равномерная решётка")
        form_u = QFormLayout(self.group_uniform)

        self.spin_N = QSpinBox()
        self.spin_N.setRange(1, 256)
        self.spin_N.setValue(16)

        self.spin_steer_x = QDoubleSpinBox()
        self.spin_steer_x.setRange(-89.0, 89.0)
        self.spin_steer_x.setValue(30.0)
        self.spin_steer_x.setSuffix("°")
        self.spin_steer_x.setDecimals(1)

        form_u.addRow("N:", self.spin_N)
        form_u.addRow("Угол θ₀:", self.spin_steer_x)

        # ── CSV ───────────────────────────────────────────────────────────
        self.group_csv = QGroupBox("CSV-файл")
        csv_layout = QVBoxLayout(self.group_csv)
        self.btn_browse = QPushButton("Выбрать файл…")
        self.btn_browse.clicked.connect(self._on_browse)
        self.lbl_csv = QLabel("файл не выбран")
        self.lbl_csv.setWordWrap(True)
        self.lbl_csv.setStyleSheet("color: gray; font-size: 10px;")
        csv_layout.addWidget(self.btn_browse)
        csv_layout.addWidget(self.lbl_csv)
        self.group_csv.hide()

        # ── Общие параметры ───────────────────────────────────────────────
        group_common = QGroupBox("Параметры")
        form_c = QFormLayout(group_common)

        self.spin_freq = QDoubleSpinBox()
        self.spin_freq.setRange(0.1, 100.0)
        self.spin_freq.setValue(3.0)
        self.spin_freq.setSuffix(" ГГц")
        self.spin_freq.setDecimals(2)

        self.combo_elem = QComboBox()
        self.combo_elem.addItems(["isotropic", "cosine", "dipole"])
        self.combo_elem.setCurrentText("cosine")

        self.spin_points = QSpinBox()
        self.spin_points.setRange(51, 1001)
        self.spin_points.setValue(181)
        self.spin_points.setSingleStep(10)

        form_c.addRow("Частота:", self.spin_freq)
        form_c.addRow("Элемент:", self.combo_elem)
        form_c.addRow("Точки θ:", self.spin_points)

        # ── Кнопка и статус ───────────────────────────────────────────────
        self.btn_calc = QPushButton("Рассчитать")
        self.btn_calc.clicked.connect(self._on_calculate)
        self.status_label = QLabel("Готов")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ctrl_layout.addWidget(group_mode)
        ctrl_layout.addWidget(self.group_uniform)
        ctrl_layout.addWidget(self.group_csv)
        ctrl_layout.addWidget(group_common)
        ctrl_layout.addWidget(self.btn_calc)
        ctrl_layout.addWidget(self.status_label)

        # --- Plots ---
        self.tabs = QTabWidget()

        self.canvas_1d = PlotCanvas()
        self.canvas_heatmap = PlotCanvas()
        self.canvas_3d = PlotCanvas(projection="3d")
        self.canvas_balloon = PlotCanvas(projection="3d")

        self.tabs.addTab(self._wrap(self.canvas_1d), "1D")
        self.tabs.addTab(self._wrap(self.canvas_heatmap), "2D Тепловая карта")
        self.tabs.addTab(self._wrap(self.canvas_3d), "3D Поверхность")
        self.tabs.addTab(self._wrap(self.canvas_balloon), "3D Шар")

        splitter.addWidget(controls)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

    # ── Вспомогательные ──────────────────────────────────────────────────

    def _wrap(self, canvas):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        return w

    def _on_mode_changed(self, use_csv: bool):
        self.group_uniform.setVisible(not use_csv)
        self.group_csv.setVisible(use_csv)

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать CSV-файл",
            str(Path(__file__).parent),
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            self._csv_path = path
            self.lbl_csv.setText(Path(path).name)
            self.lbl_csv.setStyleSheet("color: black; font-size: 10px;")

    # ── Расчёт ───────────────────────────────────────────────────────────

    def _on_calculate(self):
        if self.worker and self.worker.isRunning():
            return

        use_csv = self.chk_csv.isChecked()
        if use_csv and not self._csv_path:
            QMessageBox.warning(self, "Нет файла", "Выберите CSV-файл перед расчётом.")
            return

        params = {
            "freq": self.spin_freq.value() * 1e9,
            "elem_pattern": self.combo_elem.currentText(),
            "n_points": self.spin_points.value(),
            "csv_path": self._csv_path if use_csv else None,
            "N": self.spin_N.value(),
            "steer_x": self.spin_steer_x.value(),
        }
        self.btn_calc.setEnabled(False)
        self.status_label.setText("Расчёт…")
        self.worker = CalcWorker(params)
        self.worker.finished.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_result(self, data):
        logger.info("Rendering plots")
        self._plot_1d(data)
        self.btn_calc.setEnabled(True)
        self.status_label.setText("Готов")

    def _on_error(self, msg):
        self.btn_calc.setEnabled(True)
        self.status_label.setText("Ошибка")
        QMessageBox.critical(self, "Ошибка расчёта", msg)

    # ── Графики ──────────────────────────────────────────────────────────

    def _plot_1d(self, data):
        ax = self.canvas_1d.ax
        ax.cla()
        ax.plot(
            data["theta_deg"],
            np.clip(data["pattern_db"], -40, None),
            label=data["label"],
        )
        ax.axhline(-3, color="red", linestyle="--", label="-3 дБ")
        ax.axhline(-13, color="green", linestyle="--", label="-13 дБ")
        ax.set_xlabel("θ, градус")
        ax.set_ylabel("|F(θ)|, дБ")
        ax.set_title("Диаграмма направленности 1D")
        ax.legend()
        ax.grid(True)
        self.canvas_1d.draw()

    def _plot_heatmap(self, data):
        pat_2d = data["pat_2d"]
        theta_x = data["theta_x"]
        theta_y = data["theta_y"]
        log_p = np.clip(20 * np.log10(np.maximum(pat_2d, 1e-10)), -40, 0)
        ax = self.canvas_heatmap.ax
        ax.cla()
        im = ax.imshow(
            log_p,
            extent=[
                np.degrees(theta_y[0]),
                np.degrees(theta_y[-1]),
                np.degrees(theta_x[-1]),
                np.degrees(theta_x[0]),
            ],
            aspect="auto",
            cmap="jet",
            vmin=-40,
            vmax=0,
        )
        if not hasattr(self, "_cbar_heatmap"):
            self._cbar_heatmap = self.canvas_heatmap.fig.colorbar(im, ax=ax)
        else:
            self._cbar_heatmap.update_normal(im)
        self._cbar_heatmap.set_label("|F(θ_x, θ_y)|, дБ")
        ax.set_xlabel("θ_y, градус")
        ax.set_ylabel("θ_x, градус")
        ax.set_title("Диаграмма направленности 2D")
        self.canvas_heatmap.draw()

    def _plot_3d_surface(self, data):
        pat_2d = data["pat_2d"]
        theta_x = data["theta_x"]
        theta_y = data["theta_y"]
        log_p = np.clip(20 * np.log10(np.maximum(pat_2d, 1e-10)), -40, 0)
        TX, TY = np.meshgrid(np.degrees(theta_x), np.degrees(theta_y), indexing="ij")
        ax = self.canvas_3d.ax
        ax.cla()
        ax.plot_surface(TY, TX, log_p, cmap="jet", vmin=-40, vmax=0, antialiased=True)
        ax.set_xlabel("θ_y, °")
        ax.set_ylabel("θ_x, °")
        ax.set_zlabel("|F|, дБ")
        ax.set_title("3D Поверхность")
        self.canvas_3d.draw()

    def _plot_3d_balloon(self, data):
        pat_2d = data["pat_2d"]
        theta_x = data["theta_x"]
        theta_y = data["theta_y"]
        lin = pat_2d / pat_2d.max()
        TX_r, TY_r = np.meshgrid(theta_x, theta_y, indexing="ij")
        X = lin * np.cos(TX_r) * np.cos(TY_r)
        Y = lin * np.cos(TX_r) * np.sin(TY_r)
        Z = lin * np.sin(TX_r)
        ax = self.canvas_balloon.ax
        ax.cla()
        ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(lin), antialiased=True)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Шар (линейный масштаб)")
        self.canvas_balloon.draw()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
