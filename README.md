# Практика 3 курс

Этот проект позволяет быстро рассчитывать одномерную (в будущем и двумерную) атенную решетку.

# Как использовать

В папке репозитория нужно создать директорию `build` и зайти в нее

- `mkdir build && cd build`

Собрать проект

- `cmake ..`
- `make`

Проект собран, можно пробовать `antenna_array.ipynb` или `python/main.py`

# Тесты

Для запуска тестов нужен собранный проект (см. выше) и установленный `pytest`:

```bash
pip install pytest numpy
```

Запуск всех тестов из корня репозитория:

```bash
python -m pytest tests/ -v
```

Тесты, использующие C-библиотеку, автоматически пропускаются, если `build/libAntennaArray.so` (или `.dylib` на macOS) не собран.

# Benchmark

Бенчмарк сравнивает скорость `Calculate1DAntennaArray` через C-библиотеку и чистый Python.

Установить зависимости:

```bash
pip install pytest numpy pytest-benchmark
```

Запуск:

```bash
python -m pytest tests/test_benchmark.py -v
```

Сохранить результаты и сравнить с предыдущим запуском:

```bash
python -m pytest tests/test_benchmark.py --benchmark-save=baseline
python -m pytest tests/test_benchmark.py --benchmark-compare=baseline
```

Интерактивная версия: `benchmark.ipynb`