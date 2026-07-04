import cv2
import re
import easyocr
from ultralytics import YOLO

print("Loading YOLO...")
model = YOLO("runs/detect/runs/detect/plate_detector/weights/best.pt")
print("Model plat nomor berhasil dimuat")

print("Loading EasyOCR (pertama kali jalan akan download model, butuh internet)...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR siap")

# Webcam laptop (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak ditemukan!")
    exit()

print("Webcam aktif - tekan 'q' untuk keluar")

# --- Pengaturan ---
CONF_THRESHOLD = 0.4      # minimal confidence deteksi plat yang dianggap valid
OCR_EVERY_N_FRAMES = 10   # jalankan OCR tiap N frame saja (OCR berat, biar tidak nge-lag)

frame_count = 0
last_plate = ""           # cache nomor plat terakhir yang berhasil dibaca
last_expiry = None        # cache masa berlaku terakhir yang berhasil dibaca (bisa None)


def preprocess_for_ocr(crop):
    """Preprocessing sederhana supaya EasyOCR lebih akurat membaca plat."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.equalizeHist(gray)
    return gray


def clean_plate_text(raw_text):
    """Bersihkan hasil OCR: uppercase, hapus karakter selain huruf/angka/spasi."""
    text = raw_text.upper()
    text = re.sub(r"[^A-Z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_plate_and_expiry(cleaned_text):
    """
    Pisahkan nomor plat dari masa berlaku (format bulan-tahun di plat Indonesia,
    misal '0927' atau '09-27' yang berarti berlaku sampai 09/2027).

    Return: (nomor_plat, masa_berlaku atau None)
    """
    # Cari pola tanggal: 2 digit + (spasi/strip opsional) + 2 digit, biasanya di akhir teks
    match = re.search(r"(\d{2})\s?-?\s?(\d{2})$", cleaned_text)

    if match:
        expiry = f"{match.group(1)}-{match.group(2)}"
        plate_part = cleaned_text[:match.start()].strip()
        return plate_part, expiry

    return cleaned_text, None


while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame gagal dibaca")
        break

    frame_count += 1

    # Deteksi plat nomor
    results = model(frame, verbose=False, conf=CONF_THRESHOLD)
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Crop area plat dari frame asli
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Jalankan OCR hanya tiap N frame (biar performa tetap lancar)
        if frame_count % OCR_EVERY_N_FRAMES == 0:
            processed = preprocess_for_ocr(crop)
            ocr_result = reader.readtext(processed)

            if ocr_result:
                # Gabung semua potongan teks yang terbaca dalam satu kotak
                raw_text = " ".join([res[1] for res in ocr_result])
                cleaned = clean_plate_text(raw_text)
                if cleaned:
                    plate_part, expiry_part = split_plate_and_expiry(cleaned)
                    if plate_part:
                        last_plate = plate_part
                        last_expiry = expiry_part

        # Gambar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Label kelas + confidence (di atas kotak)
        label = f"plat_nomor {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Hasil OCR: nomor plat dan masa berlaku ditampilkan terpisah
        if last_plate:
            cv2.putText(frame, f"Plat: {last_plate}", (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if last_expiry:
            cv2.putText(frame, f"Berlaku s/d: {last_expiry}", (x1, y2 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.imshow("Deteksi Plat Nomor", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()