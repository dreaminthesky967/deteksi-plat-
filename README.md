# Sistem Deteksi dan Pengenalan Plat Nomor Kendaraan Indonesia

Sistem deteksi lokasi plat nomor kendaraan menggunakan **YOLOv8** dan pengenalan karakter otomatis menggunakan **EasyOCR**, dijalankan secara real-time melalui kamera (webcam laptop / kamera eksternal).

## Fitur

- Deteksi lokasi plat nomor kendaraan pada video real-time (YOLOv8, transfer learning dari `yolov8n.pt`)
- Pembacaan otomatis teks pada plat nomor menggunakan EasyOCR
- Pemisahan otomatis antara **nomor plat** dan **masa berlaku** (format bulan-tahun, misal `09-27`) yang tercetak pada plat
- Preprocessing crop plat (grayscale, resize, histogram equalization) untuk meningkatkan akurasi OCR

## Hasil Evaluasi Model

Model dilatih selama 50 epoch pada 411 gambar training, dievaluasi pada data validasi (117 gambar) dan data test (59 gambar) yang belum pernah dilihat model:

| Metrik      | Validation | Test  |
|-------------|-----------|-------|
| Precision   | 0.992     | 0.967 |
| Recall      | 0.974     | 1.000 |
| mAP50       | 0.994     | 0.992 |
| mAP50-95    | 0.939     | 0.933 |

## Struktur Proyek

```
├── train.py             # Script training YOLOv8
├── detect.py            # Deteksi + OCR real-time via kamera
├── evaluate_test.py     # Evaluasi model pada data test
├── requirements.txt
└── data.yaml            # Konfigurasi dataset (path perlu disesuaikan)
```

Dataset (`dataset/train`, `dataset/valid`, `dataset/test`) dan model hasil training (`runs/`, `*.pt`) **tidak disertakan** di repository ini karena ukurannya besar. Lihat bagian [Dataset](#dataset) di bawah untuk mendapatkannya.

## Dataset

Dataset yang digunakan: **"Plat nomor kendaraan Indonesia"** oleh Institut Teknologi Sumatera, tersedia di Roboflow Universe:
https://universe.roboflow.com/institut-teknologi-sumatera-di1xu/plat-nomor-kendaraan-indonesia

Download dalam format **YOLOv8**, lalu ekstrak ke folder proyek dengan struktur:
```
├── data.yaml
├── train/images, train/labels
├── valid/images, valid/labels
└── test/images, test/labels
```

Sesuaikan baris `path:` di `data.yaml` dengan lokasi folder proyek kamu.

## Instalasi

```bash
pip install -r requirements.txt
```

## Cara Pakai

### 1. Training model

```bash
python train.py
```
Model terbaik akan tersimpan di `runs/detect/plate_detector/weights/best.pt`.

### 2. Evaluasi pada data test

```bash
python evaluate_test.py
```

### 3. Deteksi + OCR real-time via kamera

Sesuaikan path model di `detect.py` (`YOLO("path/ke/best.pt")`), lalu jalankan:

```bash
python detect.py
```

Tekan `q` untuk keluar.

## Rencana Pengembangan Selanjutnya

- Dashboard web berbasis Flask untuk menampilkan hasil deteksi secara live
- Penyimpanan hasil deteksi (nomor plat, masa berlaku, waktu, foto) ke database
- Logic anti-duplikat/cooldown untuk mencegah pencatatan berulang pada kendaraan yang sama
- Validasi otomatis status plat kadaluarsa berdasarkan masa berlaku yang terbaca

## Lisensi Dataset

Dataset yang digunakan berlisensi CC BY 4.0, sesuai ketentuan dari pemilik dataset di Roboflow Universe.
