"""
Script training YOLOv8 untuk deteksi plat nomor kendaraan Indonesia.

Cara pakai:
1. Pastikan folder dataset (hasil ekstrak zip dari Roboflow) ada di lokasi yang sama
   dengan script ini, dengan struktur:
       dataset/
       ├── data.yaml
       ├── train/images, train/labels
       ├── valid/images, valid/labels
       └── test/images, test/labels
2. Install dependency:
       pip install ultralytics
3. Jalankan:
       python train.py

Setelah training selesai, model terbaik akan tersimpan di:
       runs/detect/plate_detector/weights/best.pt
"""

from ultralytics import YOLO

def main():
    # Mulai dari YOLOv8n pretrained (bukan dari nol) -> transfer learning, lebih cepat & akurat
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",   # path ke file data.yaml
        epochs=50,                  # cukup untuk dataset ~400 gambar, bisa dinaikkan kalau underfit
        imgsz=640,                  # ukuran gambar input
        batch=16,                   # turunkan ke 8/4 kalau kehabisan memori (CPU/GPU kecil)
        patience=15,                # early stopping kalau tidak ada improvement
        project="runs/detect",
        name="plate_detector",
        exist_ok=True,
    )

    # Evaluasi otomatis di data test setelah training
    metrics = model.val(data="data.yaml", split="test")
    print("\n=== Hasil evaluasi di test set ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    print("\nModel terbaik tersimpan di: runs/detect/plate_detector/weights/best.pt")
    print("Gunakan file ini untuk mengganti 'yolov8n.pt' di detect.py")

if __name__ == "__main__":
    main()
