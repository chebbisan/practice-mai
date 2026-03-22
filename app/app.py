import ctypes as ct
import logging
import platform
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
    QDoubleSpinBox,
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

from complex import complex_t
from util import (
    initialize_library,
    list_to_c_double_array,
    list_to_c_complex_array,
    calculate_delta_x,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

SPEED_OF_LIGHT = 3e8
ROOT = Path(__file__).parent.parent
_LIB_SUFFIX = {"Darwin": ".dylib", "Windows": ".dll"}.get(platform.system(), ".so")
LIB_PATH = ROOT / "build" / f"libAntennaArray{_LIB_SUFFIX}"


# ---------------------------------------------------------------------------
# Background calculation thread
# ---------------------------------------------------------------------------


class CalcWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, c_lib, params):
        super().__init__()
        self.c_lib = c_lib
        self.params = params

    def run(self):
        try:
            p = self.params
            wave_length = SPEED_OF_LIGHT / p["freq"]
            wave_num = 2 * np.pi / wave_length
            steer_x = np.radians(p["steer_x"])
            steer_y = np.radians(p["steer_y"])
            logger.info("Calculation started: N=%d, Nx=%d, Ny=%d, freq=%.2e",
                        p["N"], p["Nx"], p["Ny"], p["freq"])

            theta_1d = np.linspace(-np.pi / 2, np.pi / 2, p["n_points"])
            theta_x = np.linspace(-np.pi / 2, np.pi / 2, p["n_points"])
            theta_y = np.linspace(-np.pi / 2, np.pi / 2, p["n_points"])

            # --- 1D ---
            delta_x = calculate_delta_x(wave_length, steer_x)
            x_arr = np.array([i * delta_x - delta_x * (p["N"] - 1) / 2 for i in range(p["N"])])
            f_arr_1d = [complex_t(1, 0)] * theta_1d.size
            c_f = list_to_c_complex_array(f_arr_1d)
            c_x = list_to_c_double_array(x_arr)
            c_theta_1d = list_to_c_double_array(theta_1d)

            raw_1d = self.c_lib.Calculate1DAntennaArray(
                ct.c_int(p["N"]), ct.c_int(theta_1d.size),
                c_f, c_x, c_theta_1d, ct.c_double(wave_num),
            )
            pat_1d = np.array([
                abs(raw_1d[i].real + 1j * raw_1d[i].imag) for i in range(theta_1d.size)
            ])
            self.c_lib.FreeComplexArr(raw_1d)
            logger.debug("1D pattern done")

            # --- 2D ---
            delta_y = calculate_delta_x(wave_length, steer_y)
            x_arr2 = np.array([i * delta_x - delta_x * (p["Nx"] - 1) / 2 for i in range(p["Nx"])])
            y_arr = np.array([i * delta_y - delta_y * (p["Ny"] - 1) / 2 for i in range(p["Ny"])])
            c_f2 = list_to_c_complex_array([complex_t(1, 0)] * p["Nx"])
            c_x2 = list_to_c_double_array(x_arr2)
            c_tx = list_to_c_double_array(theta_x)
            c_f_y = list_to_c_complex_array([complex_t(1, 0)] * p["Ny"])
            c_y = list_to_c_double_array(y_arr)
            c_ty = list_to_c_double_array(theta_y)

            f_row = self.c_lib.Calculate1DAntennaArray(
                ct.c_int(p["Nx"]), ct.c_int(theta_x.size),
                c_f2, c_x2, c_tx, ct.c_double(wave_num),
            )
            f_col = self.c_lib.Calculate1DAntennaArray(
                ct.c_int(p["Ny"]), ct.c_int(theta_y.size),
                c_f_y, c_y, c_ty, ct.c_double(wave_num),
            )
            row_np = np.array([complex(f_row[i].real, f_row[i].imag) for i in range(theta_x.size)])
            col_np = np.array([complex(f_col[j].real, f_col[j].imag) for j in range(theta_y.size)])
            pat_2d = np.abs(np.outer(row_np, col_np))
            self.c_lib.FreeComplexArr(f_row)
            self.c_lib.FreeComplexArr(f_col)
            logger.debug("2D pattern done")

            self.finished.emit({
                "theta_1d": theta_1d,
                "pat_1d": pat_1d,
                "theta_x": theta_x,
                "theta_y": theta_y,
                "pat_2d": pat_2d,
            })
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
    def __init__(self, c_lib):
        super().__init__()
        self.c_lib = c_lib
        self.worker = None
        self.setWindowTitle("Antenna Array Calculator")
        self.resize(1200, 700)
        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Controls ---
        controls = QWidget()
        controls.setFixedWidth(240)
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        group = QGroupBox("Параметры")
        form = QFormLayout(group)

        self.spin_N = QSpinBox()
        self.spin_N.setRange(1, 256)
        self.spin_N.setValue(16)
        self.spin_Nx = QSpinBox()
        self.spin_Nx.setRange(1, 256)
        self.spin_Nx.setValue(16)
        self.spin_Ny = QSpinBox()
        self.spin_Ny.setRange(1, 256)
        self.spin_Ny.setValue(16)

        self.spin_freq = QDoubleSpinBox()
        self.spin_freq.setRange(0.1, 100.0)
        self.spin_freq.setValue(3.0)
        self.spin_freq.setSuffix(" ГГц")
        self.spin_freq.setDecimals(2)

        self.spin_steer_x = QDoubleSpinBox()
        self.spin_steer_x.setRange(-89.0, 89.0)
        self.spin_steer_x.setValue(30.0)
        self.spin_steer_x.setSuffix("°")
        self.spin_steer_x.setDecimals(1)

        self.spin_steer_y = QDoubleSpinBox()
        self.spin_steer_y.setRange(-89.0, 89.0)
        self.spin_steer_y.setValue(30.0)
        self.spin_steer_y.setSuffix("°")
        self.spin_steer_y.setDecimals(1)

        self.spin_points = QSpinBox()
        self.spin_points.setRange(51, 1001)
        self.spin_points.setValue(181)
        self.spin_points.setSingleStep(10)

        form.addRow("N (1D):", self.spin_N)
        form.addRow("Nx (2D):", self.spin_Nx)
        form.addRow("Ny (2D):", self.spin_Ny)
        form.addRow("Частота:", self.spin_freq)
        form.addRow("Угол θ_x₀:", self.spin_steer_x)
        form.addRow("Угол θ_y₀:", self.spin_steer_y)
        form.addRow("Точки θ:", self.spin_points)

        self.btn_calc = QPushButton("Рассчитать")
        self.btn_calc.clicked.connect(self._on_calculate)
        self.status_label = QLabel("Готов")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ctrl_layout.addWidget(group)
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

    def _wrap(self, canvas):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        return w

    def _on_calculate(self):
        if self.worker and self.worker.isRunning():
            return
        params = {
            "N": self.spin_N.value(),
            "Nx": self.spin_Nx.value(),
            "Ny": self.spin_Ny.value(),
            "freq": self.spin_freq.value() * 1e9,
            "steer_x": self.spin_steer_x.value(),
            "steer_y": self.spin_steer_y.value(),
            "n_points": self.spin_points.value(),
        }
        self.btn_calc.setEnabled(False)
        self.status_label.setText("Расчёт...")
        self.worker = CalcWorker(self.c_lib, params)
        self.worker.finished.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_result(self, data):
        logger.info("Rendering plots")
        self._plot_1d(data)
        self._plot_heatmap(data)
        self._plot_3d_surface(data)
        self._plot_3d_balloon(data)
        self.btn_calc.setEnabled(True)
        self.status_label.setText("Готов")

    def _on_error(self, msg):
        self.btn_calc.setEnabled(True)
        self.status_label.setText("Ошибка")
        QMessageBox.critical(self, "Ошибка расчёта", msg)

    def _plot_1d(self, data):
        theta = data["theta_1d"]
        pat = np.clip(20 * np.log10(np.maximum(data["pat_1d"], 1e-10)), -40, None)
        ax = self.canvas_1d.ax
        ax.cla()
        ax.plot(np.degrees(theta), pat)
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
            extent=[np.degrees(theta_y[0]), np.degrees(theta_y[-1]),
                    np.degrees(theta_x[-1]), np.degrees(theta_x[0])],
            aspect="auto", cmap="jet", vmin=-40, vmax=0,
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
    logger.info("Loading library: %s", LIB_PATH)
    try:
        c_lib = initialize_library(str(LIB_PATH))
    except OSError as e:
        print(f"Cannot load library: {e}", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    window = MainWindow(c_lib)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
