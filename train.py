from pathlib import Path
import torch
import torch_directml
from ultralytics import YOLO
import ultralytics.engine.trainer as T
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

BASE = Path(__file__).resolve().parent
DML = torch_directml.device()

# Pakai GPU AMD 
_orig_select = T.select_device
def _select(device="", *a, **k):
    if str(device) in ("", "None", "dml"):
        return DML
    return _orig_select(device, *a, **k)
T.select_device = _select

#  running di CPU
_orig_unique = torch.unique
def _unique(x, *a, **k):
    if x.device.type == "privateuseone":
        r = _orig_unique(x.cpu(), *a, **k)
        return tuple(t.to(x.device) for t in r) if isinstance(r, tuple) else r.to(x.device)
    return _orig_unique(x, *a, **k)
torch.unique = _unique
torch.Tensor.unique = lambda self, *a, **k: _unique(self, *a, **k)

# Hitung seluruh loss di CPU 
def _cpu(x):
    if torch.is_tensor(x):
        return x.cpu()
    if isinstance(x, dict):
        return {k: _cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_cpu(v) for v in x)
    return x

class CPULoss:
    def __init__(self, crit):
        self._c = crit
    def __call__(self, preds, batch):
        return self._c(_cpu(preds), _cpu(batch))
    def __getattr__(self, name):
        return getattr(self._c, name)

_orig_init = DetectionModel.init_criterion
def _init_criterion(self):
    dev = next(self.parameters()).device
    self.to("cpu")
    try:
        crit = _orig_init(self)
    finally:
        self.to(dev)
    return CPULoss(crit)
DetectionModel.init_criterion = _init_criterion

# DirectML tidak punya API memori/cache CUDA -> nonaktifkan
T.BaseTrainer._get_memory = lambda self, *a, **k: 0.0
T.BaseTrainer._clear_memory = lambda self, *a, **k: None

# Matikan validasi di trainer 
DetectionTrainer.validate = lambda self: ({}, 0.0)


def main():
    print("[INFO] Training di GPU: AMD Radeon RX 6550M (DirectML)")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(BASE / "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=8,
        workers=0,
        amp=False,
        val=False,
        plots=False,
        patience=0,
        project=str(BASE / "runs_plat"),
        name="plate_detector",
        exist_ok=True,
    )

    weights = BASE / "runs_plat" / "plate_detector" / "weights" / "last.pt"
    print(f"[INFO] Training selesai. Model: {weights}")

    try:
        m = YOLO(str(weights)).val(data=str(BASE / "data.yaml"),
                                   device="cpu", batch=8, workers=0)
        print(f"[INFO] mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")
    except Exception as e:
        print(f"[WARN] Evaluasi CPU dilewati ({e}). Model tetap tersimpan.")


if __name__ == "__main__":
    main()