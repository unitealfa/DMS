from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    from pyzbar.pyzbar import decode as zbar_decode  # type: ignore
except Exception:
    zbar_decode = None  # type: ignore

from PIL import Image, ImageSequence


REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}
MAX_SCAN_PAGES = 20
MAX_SIDE = 720
PDF_RENDER_DPI = 110
MAX_PDF_RENDER_SECONDS = 20


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _doc_key(doc_id: Any, filename: Any, idx: int) -> str:
    sid = str(doc_id or "").strip()
    if sid:
        return f"id:{sid}"
    sfn = str(filename or "").strip().lower()
    if sfn:
        return f"fn:{sfn}"
    return f"idx:{idx}"


def _build_source_path_map(ctx: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    repo_root = REPO_ROOT

    def _add(path_like: Any) -> None:
        path = str(path_like or "").strip()
        if not path:
            return
        p = Path(path)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        normalized = str(p)
        out[normalized.lower()] = normalized
        out[p.name.lower()] = normalized

    for row in _safe_list(ctx.get("PRETRAITEMENT_RESULT")):
        if isinstance(row, dict):
            _add(row.get("path"))
    for path in _safe_list(ctx.get("INPUT_FILE")):
        _add(path)
    for path in _safe_list(ctx.get("IMAGE_ONLY_FILES")):
        _add(path)
    return out


def _iter_docs(ctx: Dict[str, Any]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    docs = ctx.get("TOK_DOCS") or ctx.get("selected") or ctx.get("FINAL_DOCS") or []
    if not isinstance(docs, list):
        return []
    return [(i, doc) for i, doc in enumerate(docs) if isinstance(doc, dict)]


def _resolve_source_path(doc: Dict[str, Any], source_map: Dict[str, str]) -> str:
    filename = str(doc.get("filename") or "").strip()
    paths = _safe_list(doc.get("paths"))
    for raw in paths:
        path = str(raw or "").strip()
        if path:
            p = Path(path)
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            return str(p)
    if filename:
        return source_map.get(filename.lower()) or source_map.get(Path(filename).name.lower()) or ""
    return ""


def _pil_to_cv_bgr(img: Image.Image) -> Any:
    if cv2 is None or np is None:
        return None
    rgb = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _open_image_frames(path: Path, max_pages: int) -> List[Image.Image]:
    out: List[Image.Image] = []
    with Image.open(path) as img:
        for frame in ImageSequence.Iterator(img):
            out.append(frame.convert("RGB"))
            if len(out) >= max_pages:
                break
    return out


def _pdf_page_count(path: Path) -> int:
    try:
        proc = subprocess.run(
            ["pdfinfo", str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        text = proc.stdout.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            if line.lower().startswith("pages:"):
                return max(0, int(line.split(":", 1)[1].strip()))
    except Exception:
        pass
    return 0


def _sample_page_numbers(total_pages: int, max_pages: int) -> List[int]:
    if total_pages <= 0:
        return []
    if total_pages <= max_pages:
        return list(range(1, total_pages + 1))

    selected: List[int] = []
    head = min(4, max_pages // 3 or 1, total_pages)
    tail = min(4, max(0, max_pages - head), total_pages)

    selected.extend(range(1, head + 1))
    if tail > 0:
        selected.extend(range(max(1, total_pages - tail + 1), total_pages + 1))

    remaining = max_pages - len(set(selected))
    if remaining > 0:
        start = head + 1
        end = total_pages - tail
        if start <= end:
            span = end - start + 1
            for i in range(remaining):
                pos = start + int(round((i + 1) * span / float(remaining + 1))) - 1
                pos = max(start, min(end, pos))
                selected.append(pos)

    out = sorted({p for p in selected if 1 <= p <= total_pages})
    if len(out) > max_pages:
        out = out[:max_pages]
    return out


def _render_pdf_page(path: Path, page_number: int) -> Image.Image | None:
    with tempfile.TemporaryDirectory(prefix="dms_visual_pdf_") as tmpdir:
        prefix = str(Path(tmpdir) / f"page_{page_number}")
        cmd = [
            "pdftoppm",
            "-png",
            "-singlefile",
            "-r",
            str(PDF_RENDER_DPI),
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            str(path),
            prefix,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=MAX_PDF_RENDER_SECONDS,
            )
        except Exception:
            return None
        img_path = Path(f"{prefix}.png")
        if not img_path.exists():
            return None
        try:
            with Image.open(img_path) as img:
                return img.convert("RGB")
        except Exception:
            return None


def _render_pdf_pages(path: Path, max_pages: int) -> Tuple[List[Image.Image], List[int], int]:
    total_pages = _pdf_page_count(path)
    page_numbers = _sample_page_numbers(total_pages, max_pages) if total_pages > 0 else list(range(1, max_pages + 1))
    out: List[Image.Image] = []
    scanned: List[int] = []
    for page_number in page_numbers:
        img = _render_pdf_page(path, page_number)
        if img is None:
            continue
        out.append(img)
        scanned.append(page_number)
    return out, scanned, total_pages


def _load_doc_pages(source_path: str, max_pages: int = MAX_SCAN_PAGES) -> Tuple[List[Image.Image], List[int], int]:
    path = Path(source_path)
    if not path.exists():
        return [], [], 0
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        frames = _open_image_frames(path, max_pages)
        return frames, list(range(1, len(frames) + 1)), len(frames)
    if ext in PDF_EXTS:
        return _render_pdf_pages(path, max_pages)
    return [], [], 0


def _prepare_arrays(img: Image.Image) -> Tuple["np.ndarray", "np.ndarray", Tuple[int, int], Tuple[float, float]]:
    if np is None:
        raise RuntimeError("numpy indisponible pour detection visuelle.")
    orig_w, orig_h = img.size
    scale = 1.0
    max_dim = max(orig_w, orig_h)
    if max_dim > MAX_SIDE:
        scale = float(MAX_SIDE) / float(max_dim)
    scan_w = max(32, int(round(orig_w * scale)))
    scan_h = max(32, int(round(orig_h * scale)))
    if (scan_w, scan_h) != (orig_w, orig_h):
        scan_img = img.resize((scan_w, scan_h), Image.Resampling.BILINEAR)
    else:
        scan_img = img
    rgb = np.asarray(scan_img.convert("RGB"), dtype=np.uint8)
    gray = np.asarray(scan_img.convert("L"), dtype=np.uint8)
    sx = float(orig_w) / float(scan_w)
    sy = float(orig_h) / float(scan_h)
    return rgb, gray, (orig_w, orig_h), (sx, sy)


def _transition_density(binary: "np.ndarray") -> Tuple[float, float]:
    if np is None or binary.size <= 4:
        return 0.0, 0.0
    row_changes = np.mean(binary[:, 1:] != binary[:, :-1]) if binary.shape[1] > 1 else 0.0
    col_changes = np.mean(binary[1:, :] != binary[:-1, :]) if binary.shape[0] > 1 else 0.0
    return float(row_changes), float(col_changes)


def _score_signature(crop_rgb: "np.ndarray", crop_gray: "np.ndarray") -> float:
    if np is None:
        return 0.0
    h, w = crop_gray.shape[:2]
    if h < 8 or w < 16:
        return 0.0
    aspect = float(w) / float(max(1, h))
    dark = crop_gray < min(190, int(np.percentile(crop_gray, 62)))
    ink_ratio = float(np.mean(dark))
    row_t, col_t = _transition_density(dark)
    contrast = float(np.std(crop_gray)) / 255.0
    score = 0.0
    if aspect >= 2.2:
        score += min(0.42, (aspect - 1.8) * 0.10)
    if 0.012 <= ink_ratio <= 0.22:
        score += 0.22
    score += min(0.18, row_t * 0.75)
    score += min(0.12, contrast * 0.35)
    score -= max(0.0, col_t - 0.40) * 0.18
    return max(0.0, min(score, 1.0))


def _score_stamp(crop_rgb: "np.ndarray", crop_gray: "np.ndarray") -> float:
    if np is None:
        return 0.0
    h, w = crop_gray.shape[:2]
    if h < 16 or w < 16:
        return 0.0
    r = crop_rgb[:, :, 0].astype(np.int16)
    g = crop_rgb[:, :, 1].astype(np.int16)
    b = crop_rgb[:, :, 2].astype(np.int16)
    red_mask = (r > 90) & (r > g + 18) & (r > b + 18)
    blue_mask = (b > 80) & (b > r + 14) & (b > g + 10)
    color_ratio = float(np.mean(red_mask | blue_mask))
    dark_ratio = float(np.mean(crop_gray < 190))
    aspect = float(w) / float(max(1, h))
    square_bonus = max(0.0, 1.0 - abs(aspect - 1.0))
    score = (color_ratio * 2.2) + (dark_ratio * 0.35) + (square_bonus * 0.18)
    return max(0.0, min(score, 1.0))


def _score_qrcode(crop_rgb: "np.ndarray", crop_gray: "np.ndarray") -> float:
    if np is None:
        return 0.0
    h, w = crop_gray.shape[:2]
    if h < 20 or w < 20:
        return 0.0
    aspect = float(w) / float(max(1, h))
    if not (0.78 <= aspect <= 1.22):
        return 0.0
    thresh = int(np.percentile(crop_gray, 55))
    binary = crop_gray < thresh
    black_ratio = float(np.mean(binary))
    row_t, col_t = _transition_density(binary)
    balance = max(0.0, 1.0 - abs(black_ratio - 0.38) * 2.4)
    score = (row_t * 0.42) + (col_t * 0.42) + (balance * 0.22)
    return max(0.0, min(score, 1.0))


def _score_barcode(crop_rgb: "np.ndarray", crop_gray: "np.ndarray") -> float:
    if np is None:
        return 0.0
    h, w = crop_gray.shape[:2]
    if h < 12 or w < 30:
        return 0.0
    aspect = float(w) / float(max(1, h))
    if aspect < 1.8:
        return 0.0
    thresh = int(np.percentile(crop_gray, 55))
    binary = crop_gray < thresh
    black_ratio = float(np.mean(binary))
    row_t, col_t = _transition_density(binary)
    aspect_bonus = min(0.28, (aspect - 1.8) * 0.06)
    balance = max(0.0, 1.0 - abs(black_ratio - 0.42) * 2.1)
    score = (row_t * 0.68) + (balance * 0.18) + aspect_bonus - (col_t * 0.18)
    return max(0.0, min(score, 1.0))


def _detect_qr_barcode_decoders(img: Image.Image, page_index: int) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    width, height = img.size

    if zbar_decode is not None:
        try:
            for sym in zbar_decode(img.convert("L")):
                rect = getattr(sym, "rect", None)
                if rect is None:
                    continue
                x0 = _safe_int(getattr(rect, "left", 0))
                y0 = _safe_int(getattr(rect, "top", 0))
                w = max(1, _safe_int(getattr(rect, "width", 0)))
                h = max(1, _safe_int(getattr(rect, "height", 0)))
                x1 = x0 + w
                y1 = y0 + h
                raw_type = str(getattr(sym, "type", "") or "").strip().upper()
                det_type = "qrcode" if "QRCODE" in raw_type or "QR" == raw_type else "barcode"
                detections.append(
                    {
                        "type": det_type,
                        "score": 0.99,
                        "bbox_px": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                        "bbox_norm": {
                            "x0": round(x0 / float(max(1, width)), 6),
                            "y0": round(y0 / float(max(1, height)), 6),
                            "x1": round(x1 / float(max(1, width)), 6),
                            "y1": round(y1 / float(max(1, height)), 6),
                        },
                        "page_width": width,
                        "page_height": height,
                        "page_index": int(page_index),
                        "source": "pyzbar",
                        "decoded_type": raw_type or None,
                        "decoded_value": (
                            getattr(sym, "data", b"").decode("utf-8", errors="ignore") or None
                        ),
                    }
                )
        except Exception:
            pass

    if cv2 is not None:
        try:
            bgr = _pil_to_cv_bgr(img)
            if bgr is not None:
                detector = cv2.QRCodeDetector()
                ok, decoded_infos, points, _ = detector.detectAndDecodeMulti(bgr)
                if ok and points is not None:
                    decoded_infos = list(decoded_infos or [])
                    for idx, quad in enumerate(points):
                        if quad is None:
                            continue
                        xs = [float(pt[0]) for pt in quad]
                        ys = [float(pt[1]) for pt in quad]
                        x0 = max(0, int(round(min(xs))))
                        y0 = max(0, int(round(min(ys))))
                        x1 = min(width, int(round(max(xs))))
                        y1 = min(height, int(round(max(ys))))
                        detections.append(
                            {
                                "type": "qrcode",
                                "score": 0.98,
                                "bbox_px": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                                "bbox_norm": {
                                    "x0": round(x0 / float(max(1, width)), 6),
                                    "y0": round(y0 / float(max(1, height)), 6),
                                    "x1": round(x1 / float(max(1, width)), 6),
                                    "y1": round(y1 / float(max(1, height)), 6),
                                },
                                "page_width": width,
                                "page_height": height,
                                "page_index": int(page_index),
                                "source": "opencv-qrcode",
                                "decoded_value": decoded_infos[idx] if idx < len(decoded_infos) else None,
                            }
                        )
        except Exception:
            pass

    return detections


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(0, ax1 - ax0) * max(0, ay1 - ay0))
    area_b = float(max(0, bx1 - bx0) * max(0, by1 - by0))
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def _scan_windows(
    rgb: "np.ndarray",
    gray: "np.ndarray",
    kind: str,
    orig_size: Tuple[int, int],
    scale_xy: Tuple[float, float],
) -> List[Dict[str, Any]]:
    if np is None:
        return []
    height, width = gray.shape[:2]
    orig_w, orig_h = orig_size
    sx, sy = scale_xy

    if kind == "signature":
        sizes = [(0.52, 0.10), (0.42, 0.08), (0.34, 0.07)]
        y_min, y_max = 0.48, 0.98
        scorer = _score_signature
        threshold = 0.38
    elif kind == "stamp":
        sizes = [(0.18, 0.18), (0.22, 0.16), (0.14, 0.14)]
        y_min, y_max = 0.10, 0.95
        scorer = _score_stamp
        threshold = 0.42
    elif kind == "qrcode":
        sizes = [(0.14, 0.14), (0.18, 0.18), (0.24, 0.24)]
        y_min, y_max = 0.05, 0.95
        scorer = _score_qrcode
        threshold = 0.48
    else:
        sizes = [(0.34, 0.10), (0.28, 0.08), (0.42, 0.12)]
        y_min, y_max = 0.05, 0.95
        scorer = _score_barcode
        threshold = 0.46

    raw: List[Dict[str, Any]] = []
    for wf, hf in sizes:
        win_w = max(24, int(width * wf))
        win_h = max(18, int(height * hf))
        step_x = max(10, win_w // 3)
        step_y = max(10, win_h // 3)
        start_y = max(0, int(height * y_min))
        stop_y = min(max(0, height - win_h), int(height * y_max))
        if stop_y < start_y:
            stop_y = start_y
        for y in range(start_y, stop_y + 1, step_y):
            for x in range(0, max(1, width - win_w + 1), step_x):
                crop_gray = gray[y:y + win_h, x:x + win_w]
                crop_rgb = rgb[y:y + win_h, x:x + win_w]
                score = scorer(crop_rgb, crop_gray)
                if score < threshold:
                    continue
                x0 = int(round(x * sx))
                y0 = int(round(y * sy))
                x1 = int(round((x + win_w) * sx))
                y1 = int(round((y + win_h) * sy))
                raw.append(
                    {
                        "type": kind,
                        "score": round(float(score), 6),
                        "bbox_px": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                        "bbox_norm": {
                            "x0": round(x0 / float(max(1, orig_w)), 6),
                            "y0": round(y0 / float(max(1, orig_h)), 6),
                            "x1": round(x1 / float(max(1, orig_w)), 6),
                            "y1": round(y1 / float(max(1, orig_h)), 6),
                        },
                        "page_width": orig_w,
                        "page_height": orig_h,
                        "source": "vision-heuristics",
                    }
                )

    raw.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    kept: List[Dict[str, Any]] = []
    for cand in raw:
        bbox = cand.get("bbox_px") or {}
        a = (
            _safe_int(bbox.get("x0")),
            _safe_int(bbox.get("y0")),
            _safe_int(bbox.get("x1")),
            _safe_int(bbox.get("y1")),
        )
        if any(_bbox_iou(a, (
            _safe_int((row.get("bbox_px") or {}).get("x0")),
            _safe_int((row.get("bbox_px") or {}).get("y0")),
            _safe_int((row.get("bbox_px") or {}).get("x1")),
            _safe_int((row.get("bbox_px") or {}).get("y1")),
        )) > 0.35 for row in kept):
            continue
        kept.append(cand)
        if len(kept) >= 2:
            break
    return kept


def _detect_page_marks(img: Image.Image, page_index: int) -> List[Dict[str, Any]]:
    if np is None:
        return []
    rgb, gray, orig_size, scale_xy = _prepare_arrays(img)
    detections: List[Dict[str, Any]] = _detect_qr_barcode_decoders(img, page_index)
    for kind in ("signature", "stamp", "barcode", "qrcode"):
        for row in _scan_windows(rgb, gray, kind, orig_size, scale_xy):
            det = dict(row)
            det["page_index"] = int(page_index)
            detections.append(det)
    detections.sort(key=lambda x: (str(x.get("type") or ""), -_safe_float(x.get("score"), 0.0)))
    return detections


def _dedupe_doc_detections(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in sorted(rows, key=lambda x: float(x.get("score") or 0.0), reverse=True):
        bbox = row.get("bbox_px") or {}
        a = (
            _safe_int(bbox.get("x0")),
            _safe_int(bbox.get("y0")),
            _safe_int(bbox.get("x1")),
            _safe_int(bbox.get("y1")),
        )
        same_type = [x for x in out if str(x.get("type") or "") == str(row.get("type") or "")]
        if any(
            _bbox_iou(a, (
                _safe_int((r.get("bbox_px") or {}).get("x0")),
                _safe_int((r.get("bbox_px") or {}).get("y0")),
                _safe_int((r.get("bbox_px") or {}).get("x1")),
                _safe_int((r.get("bbox_px") or {}).get("y1")),
            )) > 0.40
            and _safe_int(r.get("page_index"), 0) == _safe_int(row.get("page_index"), 0)
            for r in same_type
        ):
            continue
        out.append(row)
    out.sort(key=lambda x: (_safe_int(x.get("page_index"), 0), str(x.get("type") or ""), -_safe_float(x.get("score"), 0.0)))
    return out


def run(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    if np is None:
        ctx["VISUAL_MARKS_DETECTIONS"] = []
        ctx["VISUAL_MARKS_DETECTIONS_100ML"] = []
        print("[visual-detection-100ml] skipped | numpy unavailable")
        return []

    source_map = _build_source_path_map(ctx)
    out: List[Dict[str, Any]] = []
    total_counts = {"signature": 0, "stamp": 0, "barcode": 0, "qrcode": 0}

    for idx, doc in _iter_docs(ctx):
        doc_id = doc.get("doc_id")
        filename = str(doc.get("filename") or f"doc_{idx}")
        source_path = _resolve_source_path(doc, source_map)
        pages, sampled_pages, total_pages = _load_doc_pages(
            source_path,
            max_pages=_safe_int(ctx.get("VISUAL_SCAN_MAX_PAGES"), MAX_SCAN_PAGES),
        )

        detections: List[Dict[str, Any]] = []
        for page_number, img in zip(sampled_pages, pages):
            detections.extend(_detect_page_marks(img, page_index=page_number))
        detections = _dedupe_doc_detections(detections)

        counts = {
            "signature": sum(1 for d in detections if str(d.get("type") or "") == "signature"),
            "stamp": sum(1 for d in detections if str(d.get("type") or "") == "stamp"),
            "barcode": sum(1 for d in detections if str(d.get("type") or "") == "barcode"),
            "qrcode": sum(1 for d in detections if str(d.get("type") or "") == "qrcode"),
        }
        for key, value in counts.items():
            total_counts[key] += int(value)

        out.append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "doc_key": _doc_key(doc_id, filename, idx),
                "engine": "visual-100ml-hybrid-heuristics-v1",
                "source_path": source_path or None,
                "pages_total": int(total_pages),
                "pages_scanned": len(pages),
                "sampled_pages": sampled_pages,
                "detections_count": len(detections),
                "has_signature": bool(counts["signature"]),
                "has_stamp": bool(counts["stamp"]),
                "has_barcode": bool(counts["barcode"]),
                "has_qrcode": bool(counts["qrcode"]),
                "counts": counts,
                "detections": detections,
            }
        )

    ctx["VISUAL_MARKS_DETECTIONS"] = out
    ctx["VISUAL_MARKS_DETECTIONS_100ML"] = out

    print(
        "[visual-detection-100ml] "
        f"docs={len(out)} | signature={total_counts['signature']} | stamp={total_counts['stamp']} | "
        f"barcode={total_counts['barcode']} | qrcode={total_counts['qrcode']}"
    )
    for row in out:
        print(
            "  - "
            f"{row.get('filename')} | pages={_safe_int(row.get('pages_scanned'), 0)}/{_safe_int(row.get('pages_total'), 0)} | "
            f"sampled={_safe_list(row.get('sampled_pages'))[:8]}{'...' if len(_safe_list(row.get('sampled_pages'))) > 8 else ''} | "
            f"signature={1 if row.get('has_signature') else 0} | "
            f"stamp={1 if row.get('has_stamp') else 0} | "
            f"barcode={1 if row.get('has_barcode') else 0} | "
            f"qrcode={1 if row.get('has_qrcode') else 0}"
        )
    return out


_CTX = globals()
run(_CTX)
