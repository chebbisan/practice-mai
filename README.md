# Практика 3 курс

Проект вычисляет диаграммы направленности одномерных и двумерных антенных решёток.
Ядро написано на C++, Python загружает скомпилированную библиотеку через `ctypes`.

## Требования

- CMake 3.28+
- C++ компилятор (gcc / clang)
- Python 3.9+

## Структура проекта

```
├── lib/                    # C++ ядро
│   ├── antenna_array.cpp/hpp   # расчёт диаграмм, волновое число, шаги
│   ├── complex.cpp/hpp         # структура complex_t
│   └── sum.cpp/hpp             # вспомогательные суммы
├── src/main.cpp            # standalone C++ исполняемый файл
├── python/
│   ├── complex.py          # зеркало C-структуры complex_t
│   ├── util.py             # загрузка библиотеки, конвертеры, Python-реализации
│   ├── main.py             # скрипт: 1D диаграмма + график
│   ├── main2d.py           # скрипт: 2D диаграмма + тепловая карта + 3D графики
│   ├── app.py              # PyQt6 GUI
│   └── config.yaml         # параметры решёток (N, Nx/Ny, freq, d, n_theta)
├── tests/
│   ├── conftest.py
│   ├── test_antenna_array.py   # 30 unit-тестов
│   └── test_benchmark.py       # бенчмарк C vs Python
├── run.sh                  # скрипт сборки + запуска (1d|2d|app|nb|bench|test)
├── antenna_array.ipynb     # интерактивная визуализация
├── benchmark.ipynb         # сравнение производительности
└── requirements.txt
```

---

## Сборка C++ библиотеки

```bash
mkdir build && cd build
cmake ..
make
```

Создаёт:
- `build/Practice` — standalone исполняемый файл
- `build/libAntennaArray.so` / `.dylib` (macOS) — библиотека для Python

---

## Виртуальное окружение

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Деактивировать:

```bash
deactivate
```

---

## Запуск

### Универсальный скрипт run.sh

`run.sh` автоматически собирает проект (если нужно) и запускает нужный режим одной командой:

```bash
./run.sh 1d      # 1D диаграмма направленности
./run.sh 2d      # 2D диаграмма (тепловая карта + 3D)
./run.sh app     # PyQt6 GUI
./run.sh nb      # Jupyter notebook
./run.sh bench   # бенчмарк
./run.sh test    # тесты
```

### Конфигурация (config.yaml)

Параметры решёток задаются в `python/config.yaml`:

```yaml
array_1d:
  N: 16           # число элементов
  freq_hz: 3.0e+9 # несущая частота, Гц
  d: null         # шаг (null → λ/(1+sin(steer_deg)))
  steer_deg: 30   # максимальный угол сканирования (при d=null)
  n_theta: 1001   # угловое разрешение

array_2d:
  Nx: 8           # элементов по x
  Ny: 32          # элементов по y
  freq_hz: 3.0e+9
  d_x: null       # шаг по x (null → λ/2)
  d_y: null       # шаг по y (null → λ/2)
  n_theta: 361    # точек на ось
```

### GUI (PyQt6)

```bash
cd python && python app.py
```

Параметры в левой панели: N элементов (1D), Nx/Ny (2D), частота, углы наведения θ_x₀/θ_y₀, количество точек θ.
Результат отображается в 4 вкладках: **1D**, **2D тепловая карта**, **3D поверхность**, **3D шар**.

### Скрипты

```bash
# 1D диаграмма направленности
cd python && python main.py

# 2D диаграмма (тепловая карта + два 3D графика)
cd python && python main2d.py
```

### Jupyter notebooks

```bash
jupyter notebook antenna_array.ipynb   # визуализация
jupyter notebook benchmark.ipynb       # сравнение производительности
```

---

## Тесты

Требуется собранный проект и активированное окружение.

```bash
python -m pytest tests/ -v
```

Тесты C-библиотеки автоматически пропускаются, если `.so` / `.dylib` не собран.

---

## Benchmark

Сравнивает скорость `Calculate1DAntennaArray` через C-библиотеку и чистый Python.

```bash
python -m pytest tests/test_benchmark.py -v
```

Сохранить результаты и сравнить с предыдущим запуском:

```bash
python -m pytest tests/test_benchmark.py --benchmark-save=baseline
python -m pytest tests/test_benchmark.py --benchmark-compare=baseline
```

Интерактивная версия: `benchmark.ipynb`
