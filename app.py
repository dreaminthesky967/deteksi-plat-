import base64
import itertools
import platform
import re
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import easyocr
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent


# KONFIGURASI SISTEM
DEBUG = False               # True: cetak hasil mentah OCR + simpan crop ke folder debug_crops/
CONF = 0.35                 # confidence deteksi untuk video
CONF_UPLOAD = 0.25          # confidence deteksi untuk upload
CONF_WEAK = 0.45            # batas deteksi lemah
IOU_NMS = 0.50
IMGSZ = 640
MAX_DET = 30
TILE_UPLOAD = True          # deteksi per potongan untuk upload gambar
TILE_VIDEO = False          # untuk video biasanya terlalu berat di CPU
TILE_MIN = 960              # gambar upload yang sisi terpanjangnya < ini di-upscale dulu
UPLOAD_MAX_W = 1920
PAD = 0.10                  # padding crop plat
MIN_CROP_W = 34             # crop lebih sempit dari 34px pada gambar jauh terlalu buram untuk OCR
MIN_CROP_H = 14             # tinggi minimum plat agar tidak menghasilkan teks halusinasi
MIN_SCORE = 0.16            # skor minimum dinaikkan agar plat buram tidak menghasilkan 'T 53 O' / 'D 4 O'
EARLY_SCORE = 0.62          # batas early-exit dinaikkan agar varian terbaik benar-benar terpilih
OCR_WARMUP_EVERY = 2
OCR_EVERY = 10
STABLE_VOTES = 5
IOU_MATCH = 0.20
MAX_MISSED = 15
SKIP = {"webcam": 0, "youtube": 2}
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-. "

FAST_VARIANTS = ["gray", "clahe", "otsu", "pca"]
FULL_VARIANTS = ["gray", "clahe", "sharp", "pca", "otsu", "adapt", "invert"]


# KLASIFIKASI HURUF & ANGKA 
CHAR_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "U": "0", "C": "0",
    "I": "1", "L": "1", "J": "1", "|": "1",
    "Z": "2",
    "E": "3",
    "A": "4", "H": "4",
    "S": "5",
    "G": "6",
    "T": "7", "Y": "7",
    "B": "8",
    "P": "9",
}

# Pemetaan karakter ke HURUF
DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "P",
}

# Kemiripan antar-huruf untuk memperbaiki Kode Wilayah 
LETTER_CONFUSION = {
    "O": "DGBQ",
    "D": "OBG",
    "B": "DG8",
    "G": "CD0",
    "C": "G",
    "Q": "ODG",
    "I": "LT1",
    "L": "IT",
    "T": "IL",
    "Z": "2",
    "S": "5",
    "A": "4",
    "E": "F",
    "F": "E",
    "K": "X",
    "X": "K",
    "V": "UW",
    "U": "VW",
    "W": "VU",
    "M": "N",
    "N": "M",
}

# Daftar resmi Kode Wilayah Plat Nomor Indonesia
PREFIXES = set("""
A B D E F G H K L M N P R S T W Z
AA AB AD AE AG BA BB BD BE BG BH BK BL BM BN BP BR
DA DB DC DD DE DG DH DK DL DM DN DP DR DS DT DW
EA EB ED KB KH KT KU PA PB PG
""".split())

SPECIAL_RE = re.compile(r"^([0-9OIZSBGLQD]{3,5})\s*[-.]\s*(\d{2}|[IVX]{1,4})$")



#  INISIALISASI MODEL
weights = next((p for p in [
    BASE / "runs_plat/plate_detector/weights/best.pt",
    BASE / "runs_plat/plate_detector/weights/last.pt"
] if p.exists()), None)

if weights is None:
    raise FileNotFoundError("Model tidak ditemukan di runs_plat/plate_detector/weights/")

print(f"[INFO] Memuat model YOLO: {weights}")
model = YOLO(str(weights))
reader = easyocr.Reader(["en"], gpu=False)
infer_lock = threading.Lock()


def dbg(*args):
    if DEBUG:
        print("[DBG]", *args)



# VERIFIKASI GEOMETRI ANGKA '1' VS '7' & '3'

def verify_digit_geometry(gray_img: np.ndarray, items: list[dict]) -> list[dict]:
    h_img, w_img = gray_img.shape[:2]
    _, binary_inv = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    for it in items:
        txt = it["txt"].upper()
        if not any(ch in txt for ch in ("7", "3", "I", "L", "T")):
            continue

        x1 = max(0, int(it["x"]))
        x2 = min(w_img, int(it.get("x2", x1 + it["h"])))
        y1 = max(0, int(it["yc"] - it["h"] * 0.5))
        y2 = min(h_img, int(it["yc"] + it["h"] * 0.5))
        roi = binary_inv[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 4:
            continue

    
        clean_t = re.sub(r"[^A-Z0-9]", "", txt)
        if len(clean_t) == 1 and clean_t in ("7", "3", "T", "I", "L"):
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
            valid_blobs = [
                stats[i] for i in range(1, num_labels)
                if stats[i, cv2.CC_STAT_HEIGHT] >= 0.45 * roi.shape[0]
            ]
            if valid_blobs:
                largest = max(valid_blobs, key=lambda s: s[cv2.CC_STAT_AREA])
                bw = largest[cv2.CC_STAT_WIDTH]
                bh = max(1, largest[cv2.CC_STAT_HEIGHT])
                aspect = bw / bh
                if aspect < 0.42:
                    it["txt"] = "1"
                    continue

        # fiks angka 1 dibaca 7
        if "7" in txt and len(clean_t) >= 2:
            col_sum = (roi > 0).mean(axis=0)
            n_chars = len(txt)
            char_w = roi.shape[1] / n_chars
            new_chars = list(txt)
            for idx, ch in enumerate(new_chars):
                if ch == "7":
                    cx1 = max(0, int(idx * char_w))
                    cx2 = min(roi.shape[1], int((idx + 1) * char_w))
                    c_roi = roi[:, cx1:cx2]
                    if c_roi.shape[1] >= 3 and c_roi.shape[0] >= 10:
                        ch_h, ch_w = c_roi.shape
                        active_cols = np.where((c_roi > 0).mean(axis=0) > 0.12)[0]
                        if len(active_cols) >= 2:
                            eff_w = active_cols[-1] - active_cols[0] + 1
                            eff_aspect = eff_w / ch_h
                            top_band = c_roi[:int(ch_h * 0.22), active_cols[0]:active_cols[-1] + 1]
                            mid_band = c_roi[int(ch_h * 0.35):int(ch_h * 0.75), active_cols[0]:active_cols[-1] + 1]
                            top_fill = float((top_band > 0).mean()) if top_band.size else 1.0
                            mid_cols = np.where((mid_band > 0).mean(axis=0) > 0.25)[0]
                            mid_w = (mid_cols[-1] - mid_cols[0] + 1) if len(mid_cols) >= 2 else eff_w

                            # Ciri mutlak angka '1':
                            # 1. Rasio lebar efektif sangat ramping (< 0.43), ATAU
                            # 2. Batang tengahnya tegak lurus sempit (mid_w / ch_h < 0.32) dan lebar atap hampir sama dengan batang
                            if eff_aspect < 0.43 or (mid_w / ch_h < 0.30 and eff_w <= mid_w * 1.45):
                                new_chars[idx] = "1"
                            elif top_fill < 0.48 and (mid_w / ch_h) < 0.36:
                                new_chars[idx] = "1"
            it["txt"] = "".join(new_chars)

    return items



# MODUL KLASIFIKASI HURUF & ANGKA SERTA PARSING PLAT
def classify_character(ch: str, expected_type: str) -> tuple[str, float]:
    ch = ch.upper()
    if expected_type == "L":
        if ch.isalpha():
            return ch, 0.0
        if ch in DIGIT_TO_LETTER:
            return DIGIT_TO_LETTER[ch], 0.45
        return ch, 2.5

    if expected_type == "D":
        if ch.isdigit():
            return ch, 0.0
        if ch in CHAR_TO_DIGIT:
            return CHAR_TO_DIGIT[ch], 0.45
        return ch, 2.5

    return ch, 3.0


def _fix_prefix(pre: str) -> tuple[str, float]:
    if pre in PREFIXES:
        return pre, 0.0

    options = []
    for ch in pre:
        alts = [ch]
        if ch in DIGIT_TO_LETTER:
            mapped = DIGIT_TO_LETTER[ch]
            alts.append(mapped)
            alts.extend(list(LETTER_CONFUSION.get(mapped, "")))
        if ch in LETTER_CONFUSION:
            alts.extend(list(LETTER_CONFUSION[ch]))
        seen = set()
        options.append([x for x in alts if not (x in seen or seen.add(x))])

    for cand in itertools.product(*options):
        c_str = "".join(cand)
        if c_str in PREFIXES:
            return c_str, 0.35

    return pre, 1.4


def _score_split(s: str, p_len: int, n_len: int) -> tuple[str, float] | None:
    total = len(s)
    t_len = total - p_len - n_len
    if p_len not in (1, 2) or not (1 <= n_len <= 4) or not (0 <= t_len <= 3):
        return None

    cost = 0.0
    prefix_chars, number_chars, suffix_chars = [], [], []

    for i in range(p_len):
        ch, c = classify_character(s[i], "L")
        prefix_chars.append(ch)
        cost += c

    for i in range(p_len, p_len + n_len):
        ch, c = classify_character(s[i], "D")
        number_chars.append(ch)
        cost += c

    for i in range(p_len + n_len, total):
        ch, c = classify_character(s[i], "L")
        suffix_chars.append(ch)
        cost += c

    num_str = "".join(number_chars)
    suf_str = "".join(suffix_chars)
    if not num_str.isdigit() or (suf_str and not suf_str.isalpha()):
        return None

    if num_str.startswith("0"):
        return None

    raw_pre = "".join(prefix_chars)
    fixed_pre, pre_cost = _fix_prefix(raw_pre)
    if not fixed_pre.isalpha():
        return None

    cost += pre_cost

    # Penalti anti-halusinasi: jika plat sangat pendek (mis. "D 4 O" atau "T 53 O"),
    # jangan izinkan ada konversi paksa huruf<->angka (seperti 0->O atau 1->T)
    if total <= 4 and cost > 0.1:
        cost += 1.5

    if t_len == 0:
        cost += 0.45

    formatted = f"{fixed_pre} {num_str} {suf_str}".strip()
    return formatted, cost


def parse_plate(raw: str) -> tuple[str, float] | None:
   #  Mengurai teks mentah OCR menjadi format standar plat Indonesia [Prefix] [Nomor] [Suffix]
    tokens = re.findall(r"[A-Z0-9]+", raw.upper())
    if not tokens:
        return None

    best = None

    def consider(res, extra_cost):
        nonlocal best
        if res is not None:
            total_cost = res[1] + extra_cost
            if best is None or total_cost < best[1]:
                best = (res[0], total_cost)

    # Jika OCR sudah memisahkan menjadi 3 kelompok (Prefix | Angka | Suffix) = sangat dipercaya
    for i in range(len(tokens) - 2):
        a, b, c = tokens[i], tokens[i + 1], tokens[i + 2]
        consider(_score_split(a + b + c, len(a), len(b)), 0.5 * (len(tokens) - 3) - 0.9)

    # Jika OCR memisahkan menjadi 2 kelompok
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        consider(_score_split(a + b, len(a), len(b)), 0.5 * (len(tokens) - 2) - 0.2)

    # Uji kombinasi pemotongan string gabungan
    joined = "".join(tokens)
    max_trim = min(2, max(0, len(joined) - 3))
    for trim_left in range(max_trim + 1):
        for trim_right in range(max_trim + 1):
            end_idx = len(joined) - trim_right
            if end_idx <= trim_left:
                continue
            sub = joined[trim_left:end_idx]
            if 3 <= len(sub) <= 9:
                for p_len in (1, 2):
                    for n_len in (1, 2, 3, 4):
                        consider(_score_split(sub, p_len, n_len), 0.65 * (trim_left + trim_right))

    return best if (best and best[1] <= 2.6) else None


def parse_special(raw: str) -> str | None:
    """Mendeteksi plat dinas"""
    cleaned = re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9\-. ]", "", raw.upper())).strip()
    m = SPECIAL_RE.match(cleaned)
    if m:
        left = "".join(CHAR_TO_DIGIT.get(c, c) for c in m.group(1))
        if left.isdigit():
            return f"{left}-{m.group(2)}"
    alnum = re.sub(r"[^A-Z0-9]", "", cleaned)
    if re.fullmatch(r"\d{6}", alnum):
        return f"{alnum[:-2]}-{alnum[-2:]}"
    return None


def parse_expiry(raw: str) -> str | None:
    """
    Klasifikasi khusus angka untuk membaca masa berlaku plat (MM-YY, contoh: 04-27 atau 08 22).
    Mendukung pemisah strip, titik, spasi, maupun 4 digit rapat.
    """
    converted = "".join(CHAR_TO_DIGIT.get(c, c) if c.isalpha() else c for c in raw.upper())
    # Cari pasangan 2 digit bulan dan 2 digit tahun (mis. "08 22", "04-27", "04.27", "0427")
    for m in re.finditer(r"(?<!\d)(\d{2})\s*[-./ ]?\s*(\d{2})(?!\d)", converted):
        mm, yy = int(m.group(1)), int(m.group(2))
        if 1 <= mm <= 12 and 20 <= yy <= 35:
            return f"{mm:02d}-{yy:02d}"
    # Fallback jika ada 1 karakter noise di tengah (mis. "04 8 27" akibat baut tengah)
    m_bolt = re.search(r"(\d{2})\s*[^0-9A-Z]?\s*[0-9A-Z]\s*[^0-9A-Z]?\s*(\d{2})", converted)
    if m_bolt:
        mm, yy = int(m_bolt.group(1)), int(m_bolt.group(2))
        if 1 <= mm <= 12 and 20 <= yy <= 35:
            return f"{mm:02d}-{yy:02d}"
    return None


def no_digit_text(raw: str) -> bool:
    al = re.sub(r"[^A-Z0-9]", "", raw.upper())
    return len(al) >= 5 and not any(c.isdigit() or c in CHAR_TO_DIGIT for c in al)


def group_lines(res) -> list[list[dict]]:
    """Mengelompokkan bounding box hasil OCR menjadi baris-baris teks dari atas ke bawah."""
    items = []
    for box, txt, conf in res:
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        items.append({
            "x": min(xs),
            "x2": max(xs),
            "yc": sum(ys) / 4.0,
            "h": max(1.0, max(ys) - min(ys)),
            "txt": txt,
            "conf": float(conf),
        })
    if not items:
        return []
    items.sort(key=lambda i: i["yc"])
    lines, cur = [], [items[0]]
    for it in items[1:]:
        mean_y = sum(c["yc"] for c in cur) / len(cur)
        mean_h = sum(c["h"] for c in cur) / len(cur)
        if abs(it["yc"] - mean_y) < 0.58 * max(it["h"], mean_h):
            cur.append(it)
        else:
            lines.append(cur)
            cur = [it]
    lines.append(cur)
    return [sorted(line, key=lambda i: i["x"]) for line in lines]


def split_expiry(lines: list[list[dict]]) -> tuple[str | None, list[list[dict]]]:
    """Memisahkan baris masa berlaku di bagian bawah plat dari baris nomor plat utama."""
    if len(lines) < 2:
        return None, lines

    last_line = lines[-1]
    txt = " ".join(i["txt"] for i in last_line)
    al = re.sub(r"[^A-Z0-9]", "", txt.upper())
    others_h = max(i["h"] for l in lines[:-1] for i in l)
    last_h = max(i["h"] for i in last_line)

    exp = parse_expiry(txt)
    if exp is not None and last_h <= 0.92 * others_h:
        return exp, lines[:-1]

    maybe_digits = sum(c.isdigit() or c in CHAR_TO_DIGIT for c in al)
    if 3 <= len(al) <= 6 and last_h < 0.72 * others_h and maybe_digits >= len(al) - 1:
        return exp, lines[:-1]

    return None, lines



# PRA-PEMROSESAN CITRA, DESKEWING & OCR
def deskew_crop(crop: np.ndarray) -> np.ndarray:
    """Meluruskan crop plat nomor yang miring berdasarkan orientasi garis horizontal."""
    h, w = crop.shape[:2]
    if w < 45 or h < 16:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=25,
        minLineLength=int(w * 0.40), maxLineGap=10
    )
    if lines is None:
        return crop
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -20.0 <= angle <= 20.0:
                angles.append(angle)
    if not angles:
        return crop
    median_angle = float(np.median(angles))
    if abs(median_angle) < 1.5:
        return crop
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), median_angle, 1.0)
    return cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def normalize_polarity(gray: np.ndarray) -> np.ndarray:
    """Menormalkan polaritas plat agar selalu berlatar terang dan berteks gelap."""
    h, w = gray.shape
    roi = gray[int(h * 0.20):int(h * 0.80), int(w * 0.08):int(w * 0.92)]
    if roi.size == 0:
        roi = gray
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = float((binary > 0).mean())
    return 255 - gray if white_ratio < 0.48 else gray


def pca_gray(crop: np.ndarray) -> np.ndarray:
    """Proyeksi warna Lab menggunakan PCA untuk memaksimalkan kontras plat berwarna."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    lab -= lab.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(lab.T) + 1e-6 * np.eye(3))
    proj = lab @ vecs[:, -1]
    lo, hi = np.percentile(proj, [2, 98])
    out = np.clip((proj - lo) / (hi - lo + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return out.reshape(crop.shape[:2])


def generate_variants(crop: np.ndarray, names: list[str]):
    """Generator berbagai varian kontras/binarisasi untuk membantu EasyOCR."""
    rectified = deskew_crop(crop)
    h = rectified.shape[0]
    # Perbesar ke tinggi ~160px agar angka masa berlaku ('08 22') di bagian bawah terbaca jelas
    scale = float(np.clip(165.0 / max(h, 1), 1.0, 6.0))
    resize_fn = lambda im: cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pad_fn = lambda im: cv2.copyMakeBorder(im, 14, 14, 16, 16, cv2.BORDER_CONSTANT, value=255)

    raw_gray = resize_fn(cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY))
    base_gray = normalize_polarity(raw_gray)
    smooth = cv2.bilateralFilter(base_gray, 7, 45, 45)
    clahe_img = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(smooth)

    for name in names:
        if name == "gray":
            img = base_gray
        elif name == "clahe":
            img = clahe_img
        elif name == "sharp":
            blur = cv2.GaussianBlur(clahe_img, (0, 0), 3)
            img = cv2.addWeighted(clahe_img, 1.7, blur, -0.7, 0)
        elif name == "pca":
            img = normalize_polarity(resize_fn(pca_gray(rectified)))
        elif name == "otsu":
            _, img = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif name == "adapt":
            img = cv2.adaptiveThreshold(
                smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 39, 9
            )
        elif name == "invert":
            img = 255 - clahe_img
        else:
            continue
        yield name, pad_fn(img)


def read_plate(crop: np.ndarray, fast: bool = False) -> tuple[str | None, str | None, float, str]:
    """Menjalankan OCR multi-varian pada satu crop plat beserta verifikasi geometri angka."""
    ch_crop, cw_crop = crop.shape[:2]
    if cw_crop < MIN_CROP_W or ch_crop < MIN_CROP_H:
        return None, None, 0.0, ""

    if DEBUG:
        d = BASE / "debug_crops"
        d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / f"{int(time.time() * 1000)}.png"), crop)

    variant_names = FAST_VARIANTS if fast else FULL_VARIANTS
    candidates = []
    raw_texts = []
    all_expiries = Counter()

    for v_name, img in generate_variants(crop, variant_names):
        ocr_res = reader.readtext(
            img,
            allowlist=ALLOW,
            detail=1,
            paragraph=False,
            mag_ratio=1.5,
            text_threshold=0.42,
            low_text=0.28,
            contrast_ths=0.10,
            adjust_contrast=0.5,
        )
        lines = group_lines(ocr_res)
        dbg(v_name, [" ".join(i["txt"] for i in l) for l in lines])
        if not lines:
            continue

        # Pisahkan baris masa berlaku SEBELUM memfilter tinggi karakter utama
        expiry, main_lines = split_expiry(lines)

        # Jika belum ketemu di baris terpisah, pindai kotak-kotak kecil di 35% bagian bawah gambar
        if expiry is None:
            img_h = img.shape[0]
            bottom_items = [
                it for l in lines for it in l
                if it["yc"] > img_h * 0.62
            ]
            if bottom_items:
                bottom_txt = " ".join(it["txt"] for it in sorted(bottom_items, key=lambda x: x["x"]))
                exp_cand = parse_expiry(bottom_txt)
                if exp_cand:
                    expiry = exp_cand

        items = [it for l in main_lines for it in l]
        if not items:
            continue

        # Saring noise baut kecil pada baris utama
        max_h = max(it["h"] for it in items)
        items = [it for it in items if it["h"] >= 0.42 * max_h]

        # Verifikasi Geometri khusus untuk memperbaiki '7'/'3' palsu kembali menjadi '1'
        items = verify_digit_geometry(img, items)

        main_txt = " ".join(it["txt"] for it in items)
        raw_texts.append(main_txt)

        special = parse_special(main_txt)
        if special:
            parsed, expiry = (special, 0.15), None
        else:
            if expiry is None:
                m = re.search(r"\b([0-9OIZSB]{2}\s*[-./ ]\s*[0-9OIZSB]{2})\s*$", main_txt.upper())
                if m:
                    possible_exp = parse_expiry(m.group(1))
                    if possible_exp:
                        expiry = possible_exp
                        main_txt = main_txt[:m.start()].strip()
            parsed = parse_plate(main_txt)

        dbg(v_name, "->", parsed, expiry)
        if expiry:
            all_expiries[expiry] += 1.0

        if not parsed:
            continue

        mean_conf = float(np.mean([it["conf"] for it in items]))
        score = mean_conf * float(np.exp(-0.70 * max(parsed[1], 0.0)))

        # Tolak halusinasi pada crop plat yang sangat kecil/jauh 
        clean_len = len(re.sub(r"\s+", "", parsed[0]))
        if (cw_crop < 55 or ch_crop < 22) and (clean_len <= 4 or score < 0.38):
            continue

        candidates.append((parsed[0], expiry, score))
        if score >= EARLY_SCORE and expiry is not None:
            break

    raw_combined = " ".join(raw_texts)
    if not candidates:
        return None, None, 0.0, raw_combined

    plate_scores = Counter()
    for p_text, _, s in candidates:
        plate_scores[p_text] += s
    best_plate = plate_scores.most_common(1)[0][0]

    expiry_scores = Counter()
    for p_text, exp_text, s in candidates:
        if p_text == best_plate and exp_text:
            expiry_scores[exp_text] += s

    best_score = max(s for p_text, _, s in candidates if p_text == best_plate)
    if best_score < MIN_SCORE:
        return None, None, 0.0, raw_combined

    best_expiry = (
        expiry_scores.most_common(1)[0][0]
        if expiry_scores
        else (all_expiries.most_common(1)[0][0] if all_expiries else None)
    )
    return best_plate, best_expiry, best_score, raw_combined

# GEOMETRI, NMS & MULTI-OBJECT TRACKER

def _inter(a, b) -> float:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(a) -> float:
    return max(0, a[2] - a[0]) * max(0, a[3] - a[1])


def iou(a, b) -> float:
    inter_area = _inter(a, b)
    union_area = _area(a) + _area(b) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def nms(dets: list[list[float]]) -> list[list[float]]:
    whole = [
        d for i, d in enumerate(dets)
        if not any(
            j != i and _area(d) < 0.80 * _area(k) and _inter(d, k) > 0.80 * _area(d)
            for j, k in enumerate(dets)
        )
    ]
    keep = []
    for d in sorted(whole, key=lambda x: x[4], reverse=True):
        if all(iou(d, k) <= 0.45 for k in keep):
            keep.append(d)
    return keep


class Track:
    _next_id = 1

    def __init__(self, box):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.missed = 0
        self.last_ocr = -999
        self.n_votes = 0
        self.rejected = False
        self.plates = Counter()
        self.expiries = Counter()

    def vote(self, plate: str, expiry: str | None, weight: float):
        for counter in (self.plates, self.expiries):
            for k in counter:
                counter[k] *= 0.95
        self.plates[plate] += weight
        if expiry:
            self.expiries[expiry] += weight
        self.n_votes += 1

    @property
    def plate(self) -> str:
        return self.plates.most_common(1)[0][0] if self.plates else ""

    @property
    def expiry(self) -> str | None:
        return self.expiries.most_common(1)[0][0] if self.expiries else None


class PlateTracker:
    def __init__(self):
        self.tracks: list[Track] = []
        self.frame = 0

    def update(self, boxes: list[tuple[int, int, int, int]]) -> list[Track]:
        self.frame += 1
        pairs = sorted(
            ((iou(t.box, b), ti, bi) for ti, t in enumerate(self.tracks) for bi, b in enumerate(boxes)),
            reverse=True,
        )
        used_t, matched = set(), {}
        for score, ti, bi in pairs:
            if score < IOU_MATCH:
                break
            if ti in used_t or bi in matched:
                continue
            used_t.add(ti)
            matched[bi] = self.tracks[ti]

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1

        active_tracks = []
        for bi, b in enumerate(boxes):
            t = matched.get(bi)
            if t is None:
                t = Track(b)
                self.tracks.append(t)
            else:
                t.box = b
                t.missed = 0
            active_tracks.append(t)

        self.tracks = [t for t in self.tracks if t.missed <= MAX_MISSED]
        return active_tracks

# PIPELINE DETEKSI YOLO & ANOTASI FRAME
def _starts(size: int, tile: int, step: int) -> list[int]:
    return [0] if size <= tile else list(range(0, size - tile, step)) + [size - tile]


def detect(frame: np.ndarray, tiled: bool = False) -> list[list[float]]:
    h, w = frame.shape[:2]
    conf_thresh = CONF_UPLOAD if tiled else CONF
    scale, img = 1.0, frame

    if tiled and max(h, w) < TILE_MIN:
        scale = min(3.0, 1280.0 / max(h, w))
        img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    H, W = img.shape[:2]
    dets = []

    def run_yolo(im, ox=0, oy=0):
        th, tw = im.shape[:2]
        res = model.predict(
            im, device="cpu", imgsz=IMGSZ, conf=conf_thresh,
            iou=IOU_NMS, max_det=MAX_DET, verbose=False
        )[0]
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            if (ox > 0 and x1 < 3) or (oy > 0 and y1 < 3) or \
               (ox + tw < W and x2 > tw - 3) or (oy + th < H and y2 > th - 3):
                continue
            dets.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy, float(b.conf[0])])

    run_yolo(img)
    if tiled and max(H, W) > IMGSZ:
        tile, step = IMGSZ, int(IMGSZ * 0.75)
        for oy in _starts(H, tile, step):
            for ox in _starts(W, tile, step):
                run_yolo(img[oy:oy + tile, ox:ox + tile], ox, oy)

    out = []
    for x1, y1, x2, y2, c in nms(dets):
        bw, bh = x2 - x1, y2 - y1
        if bh <= 0 or not (1.1 <= bw / bh <= 7.0):
            continue
        out.append([
            max(0, int(x1 / scale)),
            max(0, int(y1 / scale)),
            min(w, int(x2 / scale)),
            min(h, int(y2 / scale)),
            c,
        ])
    return out


def annotate(frame: np.ndarray, tracker: PlateTracker, force_ocr: bool = False, tiled: bool = False):
    """Mendeteksi plat, menjalankan OCR, menggambar bounding box, dan mengembalikan hasil bersih."""
    h, w = frame.shape[:2]
    with infer_lock:
        dets = detect(frame, tiled=tiled)
        boxes = [tuple(d[:4]) for d in dets]
        confs = [d[4] for d in dets]
        tracks = tracker.update(boxes)

        for t, (x1, y1, x2, y2), conf in zip(tracks, boxes, confs):
            if t.rejected:
                continue
            every = OCR_WARMUP_EVERY if t.n_votes < STABLE_VOTES else OCR_EVERY
            if (force_ocr or tracker.frame - t.last_ocr >= every) and (x2 - x1) >= MIN_CROP_W:
                t.last_ocr = tracker.frame
                px, py = int((x2 - x1) * PAD), int((y2 - y1) * PAD)
                crop = frame[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
                if crop.size:
                    plate, expiry, score, raw = read_plate(crop, fast=not force_ocr)
                    if plate:
                        t.vote(plate, expiry, score)
                    elif conf < 0.85 and no_digit_text(raw):
                        t.rejected = True

    results = []
    for t, (x1, y1, x2, y2), conf in zip(tracks, boxes, confs):
        # Pada mode upload (force_ocr=True), jangan tampilkan box yang gagal terbaca OCR
        # agar tidak muncul kotak '?' atau 'plat 0.83' tanpa nomor di UI
        if t.rejected or (force_ocr and not t.plate) or (not t.plate and conf < CONF_WEAK):
            continue

        label = f"{t.plate} {t.expiry or ''}".strip() or f"plat {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        pos = (x1, max(22, y1 - 8))
        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if t.plate:
            results.append({
                "id": t.id,
                "plate": t.plate,
                "expiry": t.expiry,
                "conf": round(conf, 2),
                "x": x1,
            })

    results.sort(key=lambda r: r["x"])
    return frame, results

#  WORKER STREAMING & SERVER FLASK
state = {
    "jpg": None, "seq": 0, "plates": [], "plate": "", "expiry": None,
    "fps": 0.0, "msg": "Pilih sumber lalu klik Mulai"
}
cfg = {"src": None, "gen": 0}


def open_source(kind: str, url: str):
    if kind == "webcam":
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(0)
    import yt_dlp
    with yt_dlp.YoutubeDL({"format": "best[height<=720]/best", "quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        return cv2.VideoCapture(info["url"])


def worker():
    cap, kind, gen, prev = None, None, -1, time.time()
    tracker = PlateTracker()
    while True:
        if gen != cfg["gen"]:
            gen = cfg["gen"]
            if cap:
                cap.release()
                cap = None
            state.update(jpg=None, plates=[], plate="", expiry=None)
            tracker = PlateTracker()
            src = cfg["src"]
            if src:
                kind = src[0]
                state["msg"] = "Membuka sumber..."
                try:
                    cap = open_source(*src)
                    if not cap.isOpened():
                        raise RuntimeError("Sumber video tidak bisa dibuka")
                    state["msg"] = "Berjalan"
                except Exception as e:
                    cap = None
                    state["msg"] = f"Gagal: {e}"
            else:
                state["msg"] = "Dihentikan"

        if cap is None:
            time.sleep(0.2)
            continue

        for _ in range(SKIP.get(kind, 0)):
            cap.grab()

        ok, frame = cap.read()
        if not ok:
            state["msg"] = "Stream berhenti atau gagal dibaca"
            cap.release()
            cap = None
            continue

        if frame.shape[1] > 960:
            frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))

        frame, found = annotate(frame, tracker, tiled=TILE_VIDEO)
        now = time.time()
        state["fps"] = 1.0 / max(now - prev, 1e-6)
        prev = now
        state["plates"] = found
        state["plate"], state["expiry"] = (
            (found[0]["plate"], found[0]["expiry"]) if found else ("", None)
        )
        state["jpg"] = cv2.imencode(".jpg", frame)[1].tobytes()
        state["seq"] += 1


threading.Thread(target=worker, daemon=True).start()

app = Flask(__name__)


@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.get("/video_feed")
def video_feed():
    def gen():
        last = -1
        while True:
            if state["jpg"] and state["seq"] != last:
                last = state["seq"]
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + state["jpg"] + b"\r\n"
            time.sleep(0.02)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.post("/source")
def source():
    d = request.get_json(force=True)
    kind, url = d.get("kind"), (d.get("url") or "").strip()
    if kind == "youtube" and not url:
        return jsonify(error="Isi link YouTube dulu"), 400
    cfg["src"] = (kind, url) if kind in SKIP else None
    cfg["gen"] += 1
    return jsonify(ok=True)


@app.get("/status")
def status():
    return jsonify(
        msg=state["msg"],
        fps=round(state["fps"], 1),
        plate=state["plate"],
        expiry=state["expiry"],
        plates=[{k: p[k] for k in ("id", "plate", "expiry", "conf")} for p in state["plates"]],
    )


@app.post("/upload")
def upload():
    cfg["src"] = None
    cfg["gen"] += 1
    f = request.files.get("file")
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR) if f else None
    if img is None:
        return jsonify(error="File bukan gambar yang valid"), 400
    if img.shape[1] > UPLOAD_MAX_W:
        img = cv2.resize(img, (UPLOAD_MAX_W, int(img.shape[0] * UPLOAD_MAX_W / img.shape[1])))
    img, found = annotate(img, PlateTracker(), force_ocr=True, tiled=TILE_UPLOAD)
    b64 = base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()
    return jsonify(
        image="data:image/jpeg;base64," + b64,
        plates=[{"plate": p["plate"], "expiry": p["expiry"]} for p in found],
    )


if __name__ == "__main__":
    print("[INFO] Buka http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True)