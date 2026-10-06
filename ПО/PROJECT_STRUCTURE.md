# Структура проекта

```text
CWS_Project_Rakhimov/
├── main.py                      # точка входа
├── requirements.txt             # зависимости
├── README.md                    # установка и демонстрация
├── run_windows.bat              # запуск под Windows
├── run_linux.sh                 # запуск под Linux
├── app/
│   ├── main.py                  # Tkinter GUI и цикл обработки
│   ├── config.py                # настройки проекта
│   ├── detector.py              # YOLO/PyTorch
│   ├── ttc.py                   # расчёт дистанции и TTC
│   ├── audio.py                 # звуковые предупреждения
│   ├── can_bus.py               # CAN-симуляция/адаптер
│   ├── logging_utils.py         # CSV + application.log
│   └── calibration.py           # калибровка OpenCV
├── config/
├── logs/
├── recordings/
├── models/
├── diagrams/                    # UML для ЛР №4
└── tests/                       # модульные тесты TTC
```
