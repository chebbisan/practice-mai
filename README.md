# Расчёт и анализ направленных свойств ФАР

Программный комплекс для расчёта диаграмм направленности (ДН), коэффициента направленного действия (КНД) и анализа параметров главного и боковых лепестков фазированных антенных решёток произвольной конфигурации: линейных (1D), плоских (2D) и пространственных (3D).

Расчёт ДН выполняется на NumPy (векторизованные операции, BLAS/SIMD). C++ библиотека сохранена для бенчмарков.

## Требования

- Python 3.9+
- CMake 3.28+ и C++ компилятор (только для бенчмарков)

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir build && cd build && cmake .. && make && cd ..

./run.sh 1d      # 1D диаграмма направленности
./run.sh 2d      # 2D диаграмма (тепловая карта + срезы)
./run.sh 3d      # 3D пространственная решётка
./run.sh app     # PyQt6 GUI
./run.sh bench   # бенчмарки (Python vs NumPy vs C++)
./run.sh test    # тесты (108 тестов)
```

## Структура проекта

```
├── lib/                        # C++ ядро (для бенчмарков)
│   ├── antenna_array.cpp/hpp   # расчёт ДН 1D/2D/3D, КНД, полные пайплайны
│   ├── complex.cpp/hpp         # структура complex_t
│   └── sum.cpp/hpp             # вспомогательные суммы
├── src/main.cpp                # standalone C++ исполняемый файл
├── app/
│   ├── common.py               # общие утилиты: константы, загрузка CSV/конфига, ДН элемента
│   ├── analysis.py             # анализ ДН: ширина луча, УБЛ, симметрия, эллиптичность
│   ├── calc_1d.py              # 1D: compute_pattern(), экспорт, визуализация
│   ├── calc_2d.py              # 2D/3D: compute_pattern_2d(), экспорт, визуализация
│   ├── calc_3d.py              # 3D: обёртка над compute_pattern_2d(z_arr=...)
│   ├── plot_2d.py              # полярная тепловая карта
│   ├── gui.py                  # PyQt6 GUI
│   ├── util.py                 # загрузка C-библиотеки, ctypes-обёртки
│   ├── complex.py              # ctypes-зеркало complex_t
│   ├── config.yaml             # параметры решёток
│   ├── input/                  # входные CSV (координаты + амплитуды)
│   └── output/                 # экспортированные ДН (CST-совместимый формат)
├── tests/
│   ├── test_antenna_array.py   # 76 unit-тестов + cross-dimensional
│   ├── test_analysis.py        # 32 теста анализа ДН
│   ├── test_benchmark.py       # бенчмарки отдельных операций
│   ├── test_benchmark_1d.py    # полный пайплайн 1D (4 реализации)
│   ├── test_benchmark_2d.py    # полный пайплайн 2D (4 реализации)
│   └── test_benchmark_3d.py    # полный пайплайн 3D (4 реализации)
├── docs/
│   ├── chapter1.md             # Глава 1: расчёт ДН и КНД ФАР
│   ├── chapter2.md             # Глава 2: инструменты анализа направленных свойств
│   ├── stuff.pdf               # учебное пособие (антенные решётки)
│   └── text.pdf                # статья Габриэльян и др., ЖРЭ 2012
├── run.sh                      # сборка + запуск
└── requirements.txt
```

## Конфигурация (config.yaml)

```yaml
array_1d:
  N: 16              # число элементов
  freq_hz: 3.0e+9    # частота, Гц
  d: null             # шаг (null → λ/(1+sin(steer_deg)))
  steer_deg: 30       # угол сканирования
  n_theta: 1001       # угловое разрешение
  element_pattern: isotropic   # isotropic | cosine | dipole
  csv_file: input/array_example.csv

array_2d:
  freq_hz: 3.0e+9
  n_theta: 1001
  n_phi: 1001
  element_pattern: isotropic
  csv_file: input/array_2d_example.csv

array_3d:
  freq_hz: 3.0e+9
  n_theta: 4001
  n_phi: 4001
  element_pattern: isotropic
  csv_file: input/array_3d_example.csv
```

### CSV входные форматы

- 1D: `x_m, amplitude_db` (2 колонки)
- 2D: `x_m, y_m, amplitude_db` (3 колонки)
- 3D: `x_m, y_m, z_m, amplitude_db` (4 колонки)

Формат определяется автоматически по числу колонок.

## Анализ ДН (analysis.py)

Модуль автоматически определяет параметры главного лепестка:
- ширина луча по -3 дБ и -10 дБ (линейная интерполяция)
- сектор по первым нулям, пространственный телесный угол
- симметрия левой/правой полуширины, коэффициент эллиптичности
- УБЛ: первого, максимальный, средний (по всем боковым лепесткам)

Работает с данными в памяти и экспортированными CSV.

## Тесты

```bash
python -m pytest tests/ -v                           # все тесты (108)
python -m pytest tests/test_antenna_array.py -v      # 76 unit-тестов
python -m pytest tests/test_analysis.py -v           # 32 теста анализа
python -m pytest tests/test_benchmark_1d.py tests/test_benchmark_2d.py tests/test_benchmark_3d.py -v  # бенчмарки
```

## Линтинг

```bash
ruff check --ignore F403,F405 .
ruff format .
clang-format -i lib/*.cpp lib/*.hpp src/*.cpp
```
