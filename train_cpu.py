from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

model = YOLO("yolov8n.pt")
model.train(
    data=str(BASE / "data.yaml"),
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu",
    workers=0,
    project=str(BASE / "runs_plat"),
    name="plate_detector",
    exist_ok=True,
)