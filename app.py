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
MIN_SCORE = 0.16            # skor minimum kandidat OCR
EARLY_SCORE = 0.65          # batas early-exit
OCR_WARMUP_EVERY = 2
OCR_EVERY = 10
STABLE_VOTES = 5
IOU_MATCH = 0.20
MAX_MISSED = 15
SKIP = {"webcam": 0, "youtube": 2}
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-. "

FAST_VARIANTS = ["gray", "unsharp", "clahe", "pca"]
FULL_VARIANTS = ["gray", "unsharp", "clahe", "sharp", "pca", "otsu", "adapt", "invert"]

#  KLASIFIKASI HURUF & ANGKA (CONFUSION MATRIX)

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

# Kemiripan antar-huruf untuk memperbaiki Kode Wilayah (Prefix)
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
    "M": "NH",
    "N": "MH",
    "H": "MN",
}

PREFIXES = set("""
A B D E F G H K L M N P R S T W Z
AA AB AD AE AG BA BB BD BE BG BH BK BL BM BN BP BR
DA DB DC DD DE DG DH DK DL DM DN DP DR DS DT DW
EA EB ED KB KH KT KU PA PB PG
""".split())

SPECIAL_RE = re.compile(r"^([0-9OIZSBGLQD]{3,5})\s*[-.]\s*(\d{2}|[IVX]{1,4})$")

# 3. INISIALISASI MODEL

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

# 4. VERIFIKASI GEOMETRI KARAKTER ('1' vs '7'/'3' dan 'M' vs 'H')
def _extract_fg_mask(roi_gray: np.ndarray) -> np.ndarray:
    """
    Menghasilkan mask biner dengan TINTA = 255 (putih) dan LATAR = 0 (hitam).
    Otomatis mendeteksi polaritas tepi ROI agar tetap akurat pada varian 'invert'.
    """
    _, bw = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Cek warna mayoritas di sepanjang garis tepi ROI (latar selalu mendominasi tepi)
    border = np.concatenate([bw[0, :], bw[-1, :], bw[:, 0], bw[:, -1]])
    if float((border > 0).mean()) >= 0.5:
        return 255 - bw  # latar putih -> balik agar tinta menjadi putih (255)
    return bw


def _tight_crop(mask: np.ndarray) -> np.ndarray | None:
    """Memotong mask biner tepat pada batas terluar tinta setelah membuang bintik noise."""
    if mask.size == 0:
        return None
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = int(areas.max())
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= max(6, int(0.08 * max_area)):
            cleaned[labels == i] = 255

    ys, xs = np.where(cleaned > 0)
    if len(ys) == 0 or len(xs) == 0:
        return None
    return cleaned[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _segment_characters(roi_fg: np.ndarray, expected_len: int) -> list[np.ndarray] | None:
    """
    Memisahkan karakter di dalam ROI menggunakan connected components yang digabung secara horizontal.
    Mengembalikan list tight-crop biner jika jumlah segmen cocok dengan expected_len, jika tidak None.
    """
    h, w = roi_fg.shape[:2]
    if h < 12 or w < 6:
        return None

    if expected_len == 1:
        tc = _tight_crop(roi_fg)
        return [tc] if tc is not None else None

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(roi_fg, connectivity=8)
    boxes = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if bh >= 0.40 * h and area >= 12:
            boxes.append([x, x + bw])

    if not boxes:
        return None

    boxes.sort(key=lambda b: b[0])
    # Gabungkan komponen yang bertumpuk secara horizontal (misalnya karakter patah vertikal)
    merged = [boxes[0]]
    for bx1, bx2 in boxes[1:]:
        px1, px2 = merged[-1]
        overlap = min(px2, bx2) - max(px1, bx1)
        min_w = max(1, min(px2 - px1, bx2 - bx1))
        if overlap > 0.35 * min_w:
            merged[-1][1] = max(px2, bx2)
        else:
            merged.append([bx1, bx2])

    if len(merged) == expected_len:
        crops = []
        for bx1, bx2 in merged:
            tc = _tight_crop(roi_fg[:, max(0, bx1):min(w, bx2)])
            if tc is None:
                return None
            crops.append(tc)
        return crops

    # Fallback: cari lembah proyeksi vertikal untuk memisahkan karakter yang sedikit bersentuhan
    col_density = (roi_fg > 0).mean(axis=0)
    cuts = [0]
    step = w / expected_len
    for k in range(1, expected_len):
        center = int(round(k * step))
        lo = max(cuts[-1] + 2, int(center - 0.28 * step))
        hi = min(w - 2, int(center + 0.28 * step))
        if lo >= hi:
            return None
        valley = lo + int(np.argmin(col_density[lo:hi]))
        # Hanya terima pemotongan jika lembah cukup tipis (< 35% tinggi) agar tidak membelah huruf
        if col_density[valley] > 0.35:
            return None
        cuts.append(valley)
    cuts.append(w)

    crops = []
    for i in range(expected_len):
        tc = _tight_crop(roi_fg[:, cuts[i]:cuts[i + 1]])
        if tc is None:
            return None
        crops.append(tc)
    return crops


def _is_one_not_seven(char_fg: np.ndarray) -> bool:
    """
    Membedakan angka '1' asli (yang salah dibaca '7') dari angka '7' asli.
    - '1' memiliki batang vertikal lurus dari atas ke bawah (max_col_fill tinggi),
      pusat massa tidak bergeser miring, dan rasio lebar/tinggi sangat ramping.
    - '7' memiliki atap horizontal lebar di atas dan kaki diagonal miring (com_shift tinggi).
    """
    h, w = char_fg.shape[:2]
    if h < 12 or w < 3:
        return False

    aspect = w / h
    fg_bool = char_fg > 0

    # 1. Cek apakah ada batang vertikal lurus kontinu dari atas sampai bawah
    col_fill = fg_bool.mean(axis=0)
    max_col_fill = float(col_fill.max())

    # 2. Cek lebar atap pada 20% bagian teratas
    top_band = fg_bool[:max(2, int(h * 0.20)), :]
    top_cols = np.where(top_band.any(axis=0))[0]
    top_span = ((top_cols[-1] - top_cols[0] + 1) / w) if len(top_cols) >= 2 else (1.0 / w)

    # 3. Cek kemiringan stroke (pergeseran pusat massa X dari bagian atas ke bagian bawah)
    upper_mid = fg_bool[int(h * 0.15):int(h * 0.40), :]
    lower_mid = fg_bool[int(h * 0.65):int(h * 0.92), :]
    _, ux = np.where(upper_mid)
    _, lx = np.where(lower_mid)
    if len(ux) == 0 or len(lx) == 0:
        return False
    com_shift = (float(ux.mean()) - float(lx.mean())) / w  # positif besar pada '7' yang miring

    # Jika miring khas '7' DAN punya atap lebar, PASTI '7' (jangan pernah diubah ke '1')
    if com_shift > 0.13 and top_span >= 0.72 and max_col_fill < 0.78:
        return False

    # Syarat ketat agar karakter dikonfirmasi sebagai angka '1':
    # Sangat ramping (< 0.34) ATAU punya batang lurus vertikal penuh (>= 0.78) tanpa kemiringan '7'
    if aspect < 0.34 and max_col_fill >= 0.68:
        return True
    if aspect < 0.42 and max_col_fill >= 0.80 and abs(com_shift) < 0.12:
        return True

    return False


def _count_horizontal_runs(band: np.ndarray) -> float:
    """Menghitung median jumlah segmen tinta horizontal pada suatu pita baris."""
    runs = []
    for row in band:
        padded = np.concatenate([[0], row.astype(np.uint8), [0]])
        transitions = int(np.sum((padded[1:] == 1) & (padded[:-1] == 0)))
        runs.append(transitions)
    return float(np.median(runs)) if runs else 0.0


def _is_m_not_h(char_fg: np.ndarray) -> bool:
    """
    Membedakan huruf 'M' asli (yang salah dibaca 'H') dari huruf 'H' asli.
    - Pada 'H', bagian dalam atas (y=6..34%) dan bagian dalam bawah (y=72..96%) sama-sama kosong.
    - Pada 'M', kedua lengan diagonal V mengisi bagian dalam kiri-atas DAN kanan-atas,
      sementara bagian dalam bawah terbuka kosong.
    """
    h, w = char_fg.shape[:2]
    if h < 14 or w < 10:
        return False

    fg = char_fg > 0
    y_u1, y_u2 = int(h * 0.06), int(h * 0.34)
    y_m1, y_m2 = int(h * 0.40), int(h * 0.62)
    y_l1, y_l2 = int(h * 0.72), int(h * 0.96)

    x_1, x_mid, x_2 = int(w * 0.24), int(w * 0.50), int(w * 0.76)
    if y_u2 <= y_u1 or y_l2 <= y_l1 or x_mid <= x_1 or x_2 <= x_mid:
        return False

    upper_left = float(fg[y_u1:y_u2, x_1:x_mid].mean())
    upper_right = float(fg[y_u1:y_u2, x_mid:x_2].mean())
    upper_inner = float(fg[y_u1:y_u2, x_1:x_2].mean())
    lower_inner = float(fg[y_l1:y_l2, x_1:x_2].mean())
    mid_inner = float(fg[y_m1:y_m2, x_1:x_2].mean())

    # Hitung jumlah potongan garis horizontal di pita atas (y = 16%..38%)
    upper_runs = _count_horizontal_runs(fg[int(h * 0.16):int(h * 0.38), :])

    # Fingerprint 1: Terdeteksi jelas 3 atau 4 garis vertikal/diagonal di bagian atas (khas 'M')
    if upper_runs >= 3.0 and upper_inner >= 0.16 and lower_inner < 0.22:
        return True

    # Fingerprint 2: Saat agak blur/kecil (kasus B 2905 KIM), lengan V mengisi kedua sisi
    # bagian dalam atas secara simetris, jauh lebih padat daripada bagian dalam bawah
    if (
        upper_left >= 0.14
        and upper_right >= 0.14
        and upper_inner >= 0.20
        and (upper_inner - lower_inner) >= 0.13
    ):
        return True

    # Fingerprint 3: Bagian atas jauh lebih penuh daripada bawah & tidak berbentuk palang tipis H murni
    if upper_inner >= 0.28 and lower_inner < 0.12 and mid_inner < 0.65:
        return True

    return False


def _is_h_not_m(char_fg: np.ndarray) -> bool:
    """Menjaga agar huruf 'H' yang jelas tidak terbalik menjadi 'M'."""
    h, w = char_fg.shape[:2]
    if h < 14 or w < 10:
        return False

    fg = char_fg > 0
    upper_inner = float(fg[int(h * 0.05):int(h * 0.30), int(w * 0.26):int(w * 0.74)].mean())
    mid_inner = float(fg[int(h * 0.40):int(h * 0.62), int(w * 0.26):int(w * 0.74)].mean())
    lower_inner = float(fg[int(h * 0.72):int(h * 0.96), int(w * 0.26):int(w * 0.74)].mean())
    upper_runs = _count_horizontal_runs(fg[int(h * 0.12):int(h * 0.30), :])

    return upper_inner < 0.07 and lower_inner < 0.07 and mid_inner > 0.35 and upper_runs <= 2.0


def verify_char_geometry(gray_img: np.ndarray, items: list[dict]) -> list[dict]:
    """
    Verifikasi geometri pasca-OCR untuk memperbaiki 3 kesalahan tersering pada plat Indonesia:
    1) Angka '1' tunggal terbaca '3' atau '7' (mis. BP 1 S)
    2) Angka '1' dalam rangkaian digit terbaca '7' (mis. H 5139 AEC) tanpa merusak '7' asli (BG 5474 QN)
    3) Huruf 'M' terbaca 'H' (mis. B 2905 KIM) tanpa merusak 'H' asli
    """
    h_img, w_img = gray_img.shape[:2]

    for it in items:
        raw_txt = it["txt"].upper()
        clean_t = re.sub(r"[^A-Z0-9]", "", raw_txt)
        if not clean_t:
            continue

        needs_digit_check = any(ch in clean_t for ch in ("7", "3", "I", "L", "T"))
        needs_letter_check = any(ch in clean_t for ch in ("H", "M"))
        if not needs_digit_check and not needs_letter_check:
            continue

        x1 = max(0, int(it["x"]) - 2)
        x2 = min(w_img, int(it.get("x2", x1 + it["h"])) + 2)
        y1 = max(0, int(it["yc"] - it["h"] * 0.55))
        y2 = min(h_img, int(it["yc"] + it["h"] * 0.55))
        roi_gray = gray_img[y1:y2, x1:x2]
        if roi_gray.size == 0 or roi_gray.shape[0] < 12 or roi_gray.shape[1] < 5:
            continue

        roi_fg = _extract_fg_mask(roi_gray)
        char_crops = _segment_characters(roi_fg, len(clean_t))
        if char_crops is None:
            continue

        # Tentukan apakah token ini dominan angka atau dominan huruf
        digit_count = sum(c.isdigit() for c in clean_t)
        is_single_digit_cand = len(clean_t) == 1 and clean_t in ("7", "3", "T", "I", "L")
        is_digit_token = (digit_count >= len(clean_t) / 2) or is_single_digit_cand
        is_letter_token = digit_count == 0

        new_chars = list(clean_t)
        changed = False

        for idx, (ch, c_fg) in enumerate(zip(clean_t, char_crops)):
            ch_h, ch_w = c_fg.shape[:2]
            aspect = ch_w / max(1, ch_h)

            # A. Verifikasi Angka '1' vs '7' / '3'
            if is_digit_token:
                if len(clean_t) == 1 and ch in ("3", "7", "T", "I", "L"):
                    if _is_one_not_seven(c_fg) or (ch == "3" and aspect < 0.38):
                        new_chars[idx] = "1"
                        changed = True
                elif ch == "7" and _is_one_not_seven(c_fg):
                    new_chars[idx] = "1"
                    changed = True

            # B. Verifikasi Huruf 'M' vs 'H' (hanya pada token huruf: Prefix / Suffix)
            if is_letter_token:
                if ch == "H" and _is_m_not_h(c_fg):
                    new_chars[idx] = "M"
                    changed = True
                elif ch == "M" and _is_h_not_m(c_fg):
                    new_chars[idx] = "H"
                    changed = True

        if changed:
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

    if total <= 4 and cost > 0.1:
        cost += 1.5

    if t_len == 0:
        cost += 0.45

    formatted = f"{fixed_pre} {num_str} {suf_str}".strip()
    return formatted, cost


def parse_plate(raw: str) -> tuple[str, float] | None:
    """Mengurai teks mentah OCR menjadi format standar plat Indonesia [Prefix] [Nomor] [Suffix]."""
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

    for i in range(len(tokens) - 2):
        a, b, c = tokens[i], tokens[i + 1], tokens[i + 2]
        consider(_score_split(a + b + c, len(a), len(b)), 0.5 * (len(tokens) - 3) - 0.9)

    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        consider(_score_split(a + b, len(a), len(b)), 0.5 * (len(tokens) - 2) - 0.2)

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
    """Mendeteksi plat dinas/khusus."""
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
    """Membaca masa berlaku plat (MM-YY) dengan toleransi pemisah spasi/titik/strip."""
    converted = "".join(CHAR_TO_DIGIT.get(c, c) if c.isalpha() else c for c in raw.upper())
    for m in re.finditer(r"(?<!\d)(\d{2})\s*[-./ ]?\s*(\d{2})(?!\d)", converted):
        mm, yy = int(m.group(1)), int(m.group(2))
        if 1 <= mm <= 12 and 20 <= yy <= 35:
            return f"{mm:02d}-{yy:02d}"
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

# 6. PRA-PEMROSESAN CITRA, DESKEWING & OCR
def deskew_crop(crop: np.ndarray) -> np.ndarray:
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
    h, w = gray.shape
    roi = gray[int(h * 0.20):int(h * 0.80), int(w * 0.08):int(w * 0.92)]
    if roi.size == 0:
        roi = gray
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = float((binary > 0).mean())
    return 255 - gray if white_ratio < 0.48 else gray


def pca_gray(crop: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    lab -= lab.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(lab.T) + 1e-6 * np.eye(3))
    proj = lab @ vecs[:, -1]
    lo, hi = np.percentile(proj, [2, 98])
    out = np.clip((proj - lo) / (hi - lo + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return out.reshape(crop.shape[:2])


def generate_variants(crop: np.ndarray, names: list[str]):
    """Generator varian citra termasuk 'unsharp' tanpa blur agar celah huruf M/W/B tetap tajam."""
    rectified = deskew_crop(crop)
    h = rectified.shape[0]
    scale = float(np.clip(170.0 / max(h, 1), 1.0, 6.0))
    resize_fn = lambda im: cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pad_fn = lambda im: cv2.copyMakeBorder(im, 14, 14, 16, 16, cv2.BORDER_CONSTANT, value=255)

    raw_gray = resize_fn(cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY))
    base_gray = normalize_polarity(raw_gray)
    # Penajaman langsung tanpa bilateral blur agar lekukan V di dalam huruf 'M' tidak menyatu
    unsharp_img = cv2.addWeighted(base_gray, 1.9, cv2.GaussianBlur(base_gray, (0, 0), 2.0), -0.9, 0)
    smooth = cv2.bilateralFilter(base_gray, 5, 35, 35)
    clahe_img = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(smooth)

    for name in names:
        if name == "gray":
            img = base_gray
        elif name == "unsharp":
            img = unsharp_img
        elif name == "clahe":
            img = clahe_img
        elif name == "sharp":
            blur = cv2.GaussianBlur(clahe_img, (0, 0), 2.5)
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
    """Menjalankan OCR multi-varian pada satu crop plat beserta verifikasi geometri karakter."""
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

        expiry, main_lines = split_expiry(lines)

        if expiry is None:
            img_h = img.shape[0]
            bottom_items = [it for l in lines for it in l if it["yc"] > img_h * 0.62]
            if bottom_items:
                bottom_txt = " ".join(it["txt"] for it in sorted(bottom_items, key=lambda x: x["x"]))
                exp_cand = parse_expiry(bottom_txt)
                if exp_cand:
                    expiry = exp_cand

        items = [it for l in main_lines for it in l]
        if not items:
            continue

        max_h = max(it["h"] for it in items)
        items = [it for it in items if it["h"] >= 0.42 * max_h]

        # Verifikasi geometri presisi untuk '1' vs '7'/'3' dan 'M' vs 'H'
        items = verify_char_geometry(img, items)

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

# 8. PIPELINE DETEKSI YOLO & ANOTASI FRAME
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

# WORKER STREAMING & SERVER FLASK
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