import re
import time
from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

# --- Load model YOLO (hasil train.py: last.pt) ---
candidates = [
    BASE / "runs_plat/plate_detector/weights/best.pt",
    BASE / "runs_plat/plate_detector/weights/last.pt",
]
model_path = next((p for p in candidates if p.exists()), None)
if model_path is None:
    raise FileNotFoundError("Model tidak ditemukan di runs_plat/plate_detector/weights/")

print(f"Loading YOLO dari: {model_path}")
model = YOLO(str(model_path))
print("Model plat nomor berhasil dimuat (inferensi di CPU)")

# --- Load EasyOCR ---
print("Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=False)
print("EasyOCR siap")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam tidak ditemukan!")
    raise SystemExit

print("Kamera aktif - tekan 'q' untuk keluar")

CONF_THRESHOLD = 0.4
OCR_EVERY_N_FRAMES = 10
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "

frame_count = 0
last_plate, last_expiry = "", None
prev_time = time.time()


def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return cv2.equalizeHist(gray)


def clean_plate_text(raw):
    text = re.sub(r"[^A-Z0-9 ]", "", raw.upper())
    return re.sub(r"\s+", " ", text).strip()


def split_plate_and_expiry(text):
    m = re.search(r"(\d{2})\s?-?\s?(\d{2})$", text)
    if m:
        return text[:m.start()].strip(), f"{m.group(1)}-{m.group(2)}"
    return text, None


while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame gagal dibaca")
        break

    frame_count += 1
    now = time.time()
    fps = 1.0 / (now - prev_time) if now > prev_time else 0
    prev_time = now

    results = model.predict(frame, device="cpu", imgsz=640,
                            conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        last_plate, last_expiry = "", None

    h, w = frame.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        if frame_count % OCR_EVERY_N_FRAMES == 0:
            ocr = reader.readtext(preprocess_for_ocr(crop), allowlist=ALLOW)
            if ocr:
                cleaned = clean_plate_text(" ".join(r[1] for r in ocr))
                if cleaned:
                    plate, expiry = split_plate_and_expiry(cleaned)
                    if plate:
                        last_plate, last_expiry = plate, expiry

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"plat_nomor {conf:.2f}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if last_plate:
            cv2.putText(frame, f"Plat: {last_plate}", (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if last_expiry:
            cv2.putText(frame, f"Berlaku s/d: {last_expiry}", (x1, y2 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.putText(frame, f"CPU | FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Deteksi Plat Nomor", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()