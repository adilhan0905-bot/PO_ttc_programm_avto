from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from app.audio import play_alert
from app.calibration import CameraCalibrator
from app.can_bus import CanBusAdapter
from app.config import BASE_DIR, LOG_DIR, RECORDINGS_DIR, Settings, ensure_dirs
from app.detector import Detection, ObjectDetector
from app.logging_utils import log_warning, setup_logging


class CWSApp:
    def __init__(self, root: tk.Tk) -> None:
        ensure_dirs()
        setup_logging()
        self.root = root
        self.root.title("CWS — Автономная система предупреждения о столкновениях")
        self.root.geometry("1400x850")
        self.root.minsize(1100, 700)

        self.settings = Settings.load()
        self.detector: ObjectDetector | None = None
        self.can = CanBusAdapter(enabled=self.settings.can_enabled, simulate=True)

        self.capture: cv2.VideoCapture | None = None
        self.capture_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.last_frame = None
        self.video_fps = 30.0
        self.source_name = "Нет источника"
        self.record_writer = None
        self.is_recording = False
        self.last_alert_time = 0.0
        self.last_logged_signature = None
        self.frame_counter = 0
        self.current_detections: list[Detection] = []

        self.status_var = tk.StringVar(value="Готово. Выберите видеофайл или камеру.")
        self.ttc_var = tk.StringVar(value="—")
        self.distance_var = tk.StringVar(value="—")
        self.object_var = tk.StringVar(value="—")
        self.fps_var = tk.StringVar(value="0.0")
        self.source_var = tk.StringVar(value="Нет источника")
        self.level_var = tk.StringVar(value="SAFE")
        self.speed_var = tk.StringVar(value=f"{self.settings.vehicle_speed_kmh:.0f}")
        self.model_var = tk.StringVar(value=self.settings.model_path or "yolov8n.pt")
        self.sensitivity_var = tk.StringVar(value=str(int(self.settings.sensitivity_seconds)))
        self.mute_var = tk.BooleanVar(value=not self.settings.sound_enabled)

        self._configure_style()
        self._build_ui()
        self._load_model_async()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self._poll_results)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", padding=(10, 7), font=("Segoe UI", 10))
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
        style.configure("Metric.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=12)
        header.pack(fill="x")
        ttk.Label(header, text="CWS — Collision Warning System", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Учебный проект | Python + OpenCV + PyTorch", padding=(20, 0)).pack(side="left")

        controls = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        controls.pack(fill="x")
        ttk.Button(controls, text="Открыть видео", command=self.open_video).pack(side="left", padx=3)
        ttk.Button(controls, text="Камера", command=self.open_camera).pack(side="left", padx=3)
        ttk.Button(controls, text="Старт", command=self.start).pack(side="left", padx=3)
        ttk.Button(controls, text="Стоп", command=self.stop).pack(side="left", padx=3)
        ttk.Button(controls, text="Запись", command=self.toggle_recording).pack(side="left", padx=3)
        ttk.Button(controls, text="Самодиагностика", command=self.run_diagnostics).pack(side="left", padx=3)
        ttk.Button(controls, text="Калибровка", command=self.calibrate_camera).pack(side="left", padx=3)
        ttk.Button(controls, text="Загрузить модель", command=self.load_model).pack(side="left", padx=3)

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        video_frame = ttk.LabelFrame(main, text="Видеопоток", padding=8)
        video_frame.pack(side="left", fill="both", expand=True)
        self.video_label = ttk.Label(video_frame, text="Нет кадра", anchor="center")
        self.video_label.pack(fill="both", expand=True)

        sidebar = ttk.Frame(main, width=340)
        sidebar.pack(side="right", fill="y", padx=(12, 0))
        sidebar.pack_propagate(False)

        metrics = ttk.LabelFrame(sidebar, text="Состояние системы", padding=12)
        metrics.pack(fill="x")
        self._metric(metrics, "Уровень", self.level_var)
        self._metric(metrics, "TTC", self.ttc_var, suffix="с")
        self._metric(metrics, "Дистанция", self.distance_var, suffix="м")
        self._metric(metrics, "Объект", self.object_var)
        self._metric(metrics, "FPS", self.fps_var)
        self._metric(metrics, "Источник", self.source_var)

        settings = ttk.LabelFrame(sidebar, text="Параметры", padding=12)
        settings.pack(fill="x", pady=10)
        self._setting_row(settings, "Скорость, км/ч", self.speed_var)
        ttk.Label(settings, text="Чувствительность TTC").pack(anchor="w", pady=(10, 2))
        sensitivity = ttk.Combobox(settings, textvariable=self.sensitivity_var, values=["1", "2", "3"], state="readonly")
        sensitivity.pack(fill="x")
        sensitivity.bind("<<ComboboxSelected>>", self._save_settings)
        ttk.Checkbutton(settings, text="Без звука", variable=self.mute_var).pack(anchor="w", pady=8)

        model_box = ttk.LabelFrame(sidebar, text="Нейросеть", padding=12)
        model_box.pack(fill="x")
        ttk.Label(model_box, textvariable=self.model_var, wraplength=290).pack(anchor="w")

        log_frame = ttk.LabelFrame(sidebar, text="Последнее событие", padding=12)
        log_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.event_text = tk.Text(log_frame, height=12, wrap="word", state="disabled", font=("Consolas", 9))
        self.event_text.pack(fill="both", expand=True)

        status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=6)
        status.pack(side="bottom", fill="x")

    @staticmethod
    def _metric(parent, name: str, var: tk.StringVar, suffix: str = "") -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=name).pack(side="left")
        ttk.Label(row, textvariable=var, style="Metric.TLabel").pack(side="right")
        if suffix:
            ttk.Label(row, text=suffix).pack(side="right", padx=(4, 0))

    @staticmethod
    def _setting_row(parent, name: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=name).pack(anchor="w", pady=(0, 2))
        entry = ttk.Entry(parent, textvariable=var)
        entry.pack(fill="x")

    def _save_settings(self, _event=None) -> None:
        try:
            speed = max(0.0, float(self.speed_var.get().replace(",", ".")))
            sensitivity = float(self.sensitivity_var.get())
            self.settings.vehicle_speed_kmh = speed
            self.settings.sensitivity_seconds = sensitivity
            self.settings.sound_enabled = not self.mute_var.get()
            self.settings.model_path = self.model_var.get() if self.model_var.get() != "yolov8n.pt" else self.settings.model_path
            self.settings.save()
            self.status_var.set("Параметры сохранены.")
        except ValueError:
            self.status_var.set("Ошибка: скорость должна быть числом.")

    def _load_model_async(self) -> None:
        path = self.settings.model_path.strip() or "yolov8n.pt"
        self.model_var.set(path)
        self.status_var.set(f"Загрузка модели {path}…")
        threading.Thread(target=self._load_model_worker, args=(path,), daemon=True).start()

    def _load_model_worker(self, path: str) -> None:
        try:
            self.detector = ObjectDetector(path, self.settings.confidence, self.settings.hfov_deg)
            self.root.after(0, lambda: self.status_var.set(f"Модель готова: {path} ({self.detector.device})."))
        except Exception as exc:
            logging.exception("Model load failed")
            self.root.after(0, lambda: self.status_var.set(f"Не удалось загрузить модель: {exc}"))

    def load_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите нейросетевую модель",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
            initialdir=str(BASE_DIR / "models"),
        )
        if not path:
            return
        self.model_var.set(path)
        self.settings.model_path = path
        self.settings.save()
        self.status_var.set("Загрузка выбранной модели…")
        threading.Thread(target=self._load_model_worker, args=(path,), daemon=True).start()

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите видеопоток",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.stop()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть видеофайл.")
            return
        self.capture = cap
        self.source_name = Path(path).name
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.source_var.set(self.source_name)
        self.status_var.set(f"Открыт файл: {self.source_name}")
        self.start()

    def open_camera(self) -> None:
        self.stop()
        cap = cv2.VideoCapture(self.settings.camera_index)
        if not cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру.")
            return
        self.capture = cap
        self.source_name = f"Камера #{self.settings.camera_index}"
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.source_var.set(self.source_name)
        self.status_var.set("Камера открыта.")
        self.start()

    def start(self) -> None:
        if self.capture is None:
            self.status_var.set("Сначала выберите видеофайл или камеру.")
            return
        if self.detector is None:
            self.status_var.set("Нейросеть ещё загружается.")
            return
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._processing_loop, daemon=True)
        self.worker.start()
        self.status_var.set("Обработка видеопотока запущена.")

    def stop(self) -> None:
        self.stop_event.set()
        if self.record_writer is not None:
            self.record_writer.release()
            self.record_writer = None
        self.is_recording = False

    def _processing_loop(self) -> None:
        prev_time = time.perf_counter()
        while not self.stop_event.is_set():
            with self.capture_lock:
                cap = self.capture
                if cap is None:
                    break
                ok, frame = cap.read()
            if not ok:
                self.root.after(0, lambda: self.status_var.set("Видеопоток завершён."))
                break
            now = time.perf_counter()
            instant_fps = 1.0 / max(1e-3, now - prev_time)
            prev_time = now
            fps = 0.8 * self.video_fps + 0.2 * instant_fps if self.video_fps else instant_fps
            self.video_fps = max(1.0, min(120.0, fps))

            try:
                speed = max(0.0, float(self.speed_var.get().replace(",", ".")))
                threshold = float(self.sensitivity_var.get())
            except ValueError:
                speed, threshold = 30.0, 2.0

            try:
                detections = self.detector.infer(frame, speed, threshold, self.video_fps) if self.detector else []
                annotated = self._annotate(frame, detections, threshold)
                item = (annotated, detections, self.video_fps)
                try:
                    self.frame_queue.put_nowait(item)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put_nowait(item)
            except Exception as exc:
                logging.exception("Frame processing error")
                self.root.after(0, lambda e=str(exc): self.status_var.set(f"Ошибка обработки кадра: {e}"))
                time.sleep(0.05)

    def _annotate(self, frame, detections: list[Detection], threshold: float):
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            thickness = 3 if det.level == "CRITICAL" else 2
            color = (0, 0, 255) if det.level == "CRITICAL" else (0, 165, 255) if det.level == "WARNING" else (0, 200, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            parts = [f"{det.class_name} {det.confidence:.2f}"]
            if det.distance_m is not None:
                parts.append(f"{det.distance_m:.1f}m")
            if det.ttc_s is not None:
                parts.append(f"TTC {det.ttc_s:.2f}s")
            label = " | ".join(parts)
            cv2.rectangle(annotated, (x1, max(0, y1 - 26)), (min(annotated.shape[1] - 1, x2 + 3), y1), color, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        level = "SAFE"
        if any(d.level == "CRITICAL" for d in detections):
            level = "CRITICAL"
        elif any(d.level == "WARNING" for d in detections):
            level = "WARNING"
        banner = {
            "SAFE": "СОСТОЯНИЕ: НОРМА",
            "WARNING": "ВНИМАНИЕ: ОПАСНОЕ СБЛИЖЕНИЕ",
            "CRITICAL": "ОПАСНОСТЬ: ВОЗМОЖНО СТОЛКНОВЕНИЕ",
        }[level]
        cv2.putText(annotated, banner, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if level == "SAFE" else (0, 165, 255) if level == "WARNING" else (0, 0, 255), 2, cv2.LINE_AA)
        return annotated

    def _poll_results(self) -> None:
        try:
            frame, detections, fps = self.frame_queue.get_nowait()
            self.last_frame = frame
            self.current_detections = detections
            self.fps_var.set(f"{fps:.1f}")
            self._update_metrics(detections)
            self._handle_alerts(detections)
            self._show_frame(frame)
            if self.is_recording and self.record_writer is not None:
                self.record_writer.write(frame)
        except queue.Empty:
            pass
        self.root.after(30, self._poll_results)

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w = max(700, self.video_label.winfo_width())
        max_h = max(500, self.video_label.winfo_height())
        scale = min(max_w / image.width, max_h / image.height)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

    def _update_metrics(self, detections: list[Detection]) -> None:
        dangerous = [d for d in detections if d.ttc_s is not None]
        if not dangerous:
            self.level_var.set("SAFE")
            self.ttc_var.set("—")
            self.distance_var.set("—")
            self.object_var.set("—")
            return
        det = min(dangerous, key=lambda d: d.ttc_s if d.ttc_s is not None else 9999)
        self.level_var.set(det.level)
        self.ttc_var.set("—" if det.ttc_s is None else f"{det.ttc_s:.2f}")
        self.distance_var.set("—" if det.distance_m is None else f"{det.distance_m:.2f}")
        self.object_var.set(det.class_name)

    def _handle_alerts(self, detections: list[Detection]) -> None:
        critical = [d for d in detections if d.level == "CRITICAL" and d.ttc_s is not None]
        warning = [d for d in detections if d.level == "WARNING" and d.ttc_s is not None]
        chosen = min(critical or warning, key=lambda d: d.ttc_s if d.ttc_s is not None else 9999) if (critical or warning) else None
        if chosen is None:
            return

        now = time.time()
        signature = (chosen.class_name, chosen.level, round(chosen.ttc_s or 0, 1))
        if self.last_logged_signature != signature or now - self.last_alert_time > 2.0:
            log_warning(chosen.class_name, chosen.confidence, chosen.distance_m or 0, chosen.ttc_s, chosen.level)
            self._append_event(
                f"{datetime.now():%H:%M:%S} | {chosen.level}\n"
                f"Объект: {chosen.class_name}, дистанция: {chosen.distance_m:.1f} м, TTC: {chosen.ttc_s:.2f} с\n"
            )
            self.last_logged_signature = signature

        if now - self.last_alert_time >= 1.0:
            if not self.mute_var.get():
                play_alert(chosen.level)
            if chosen.level == "CRITICAL" and self.settings.can_enabled and chosen.ttc_s is not None:
                self.can.send_critical(float(self.speed_var.get()), chosen.ttc_s)
            self.last_alert_time = now

    def _append_event(self, text: str) -> None:
        self.event_text.configure(state="normal")
        self.event_text.insert("end", text + "\n")
        self.event_text.see("end")
        self.event_text.configure(state="disabled")

    def toggle_recording(self) -> None:
        if not self.is_recording:
            if self.last_frame is None:
                self.status_var.set("Нет кадра для начала записи.")
                return
            height, width = self.last_frame.shape[:2]
            path = RECORDINGS_DIR / f"cws_{datetime.now():%Y%m%d_%H%M%S}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.record_writer = cv2.VideoWriter(str(path), fourcc, max(5.0, self.video_fps), (width, height))
            if not self.record_writer.isOpened():
                self.record_writer = None
                messagebox.showerror("Ошибка", "Не удалось создать файл записи.")
                return
            self.is_recording = True
            self.status_var.set(f"Идёт запись: {path.name}")
        else:
            if self.record_writer is not None:
                self.record_writer.release()
            self.record_writer = None
            self.is_recording = False
            self.status_var.set("Запись остановлена.")

    def run_diagnostics(self) -> None:
        checks = []
        checks.append(("OpenCV", cv2.__version__))
        checks.append(("Модель", "OK" if self.detector is not None else "ОШИБКА"))
        checks.append(("Каталог логов", "OK" if LOG_DIR.exists() else "ОШИБКА"))
        checks.append(("Видеопоток", "OK" if self.capture is not None else "не выбран"))
        checks.append(("CAN", "симуляция" if self.settings.can_enabled else "выкл."))
        report = "\n".join(f"{name}: {value}" for name, value in checks)
        self._append_event("САМОТЕСТ:\n" + report)
        messagebox.showinfo("Самодиагностика", report)

    def calibrate_camera(self) -> None:
        folder = filedialog.askdirectory(title="Выберите папку с изображениями шахматной доски")
        if not folder:
            return
        paths = [str(p) for p in sorted(Path(folder).glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        if not paths:
            messagebox.showwarning("Калибровка", "В папке нет изображений JPG/PNG/BMP.")
            return
        output = BASE_DIR / "config" / "calibration.json"
        ok, text = CameraCalibrator().calibrate_from_images(paths, str(output))
        self._append_event("КАЛИБРОВКА:\n" + text)
        if ok:
            messagebox.showinfo("Калибровка", text)
        else:
            messagebox.showwarning("Калибровка", text)

    def close(self) -> None:
        self.stop_event.set()
        if self.record_writer is not None:
            self.record_writer.release()
        if self.capture is not None:
            self.capture.release()
        self.settings.sound_enabled = not self.mute_var.get()
        self.settings.save()
        self.root.destroy()


def main() -> None:
    ensure_dirs()
    root = tk.Tk()
    app = CWSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
