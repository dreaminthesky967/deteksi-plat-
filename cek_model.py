from pathlib import Path
from ultralytics import YOLO

B = Path(__file__).resolve().parent
w = next(p for p in [B/"runs_plat/plate_detector/weights/best.pt", B/"runs_plat/plate_detector/weights/last.pt"] if p.exists())
print("Model:", w)
m = YOLO(str(w))
for im in sorted((B/"valid/images").glob("*.jpg"))[:10]:
    r = m.predict(str(im), device="cpu", conf=0.05, verbose=False)[0]
    print(im.name[:30], [round(float(c), 2) for c in r.boxes.conf])