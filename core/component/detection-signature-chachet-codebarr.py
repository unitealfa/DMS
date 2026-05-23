from __future__ import annotations

import base64
import io
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

try:
    from pipeline.file_resolution import resolve_runtime_input_path
except Exception:
    def resolve_runtime_input_path(path: Path, repo_root: Path) -> Path:
        return path


REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}
MAX_SCAN_PAGES = 20
MAX_SIDE = 720
PDF_RENDER_DPI = 110
MAX_PDF_RENDER_SECONDS = 20
SIGNATURE_MAX_PER_PAGE = 1
PAGE_PREVIEW_MAX_SIDE = 900


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


def _image_to_data_url(img: Image.Image, *, max_side: int = PAGE_PREVIEW_MAX_SIDE, quality: int = 72) -> str | None:
    try:
        preview = img.convert("RGB")
        width, height = preview.size
        max_dim = max(width, height)
        if max_dim > max_side:
            ratio = float(max_side) / float(max_dim)
            preview = preview.resize(
                (max(32, int(round(width * ratio))), max(32, int(round(height * ratio)))),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        preview.save(buf, format="JPEG", quality=quality, optimize=True)
        payload = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"
    except Exception:
        return None


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
        p = resolve_runtime_input_path(p, repo_root)
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
    path = resolve_runtime_input_path(path, REPO_ROOT)
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
    path = resolve_runtime_input_path(Path(source_path), REPO_ROOT)
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


def _active_segments(
    signal: "np.ndarray",
    threshold: float,
    *,
    min_len: int = 1,
    max_gap: int = 0,
) -> List[Tuple[int, int]]:
    if np is None:
        return []
    active = np.asarray(signal >= threshold, dtype=bool)
    segments: List[Tuple[int, int]] = []
    start = -1
    gap = 0
    for idx, flag in enumerate(active.tolist()):
        if flag:
            if start < 0:
                start = idx
            gap = 0
            continue
        if start < 0:
            continue
        if gap < max_gap:
            gap += 1
            continue
        end = idx - gap - 1
        if end >= start and (end - start + 1) >= min_len:
            segments.append((start, end))
        start = -1
        gap = 0
    if start >= 0:
        end = len(active) - 1
        if (end - start + 1) >= min_len:
            segments.append((start, end))
    return segments


def _mask_components(mask: "np.ndarray", *, min_area: int = 1) -> List[Dict[str, int]]:
    if np is None or mask.size == 0:
        return []
    binary = np.asarray(mask, dtype=bool)
    h, w = binary.shape[:2]
    visited = np.zeros((h, w), dtype=bool)
    ys, xs = np.nonzero(binary)
    components: List[Dict[str, int]] = []
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        visited[sy, sx] = True
        stack = [(sy, sx)]
        area = 0
        y0 = y1 = sy
        x0 = x1 = sx
        while stack:
            y, x = stack.pop()
            area += 1
            if y < y0:
                y0 = y
            if y > y1:
                y1 = y
            if x < x0:
                x0 = x
            if x > x1:
                x1 = x
            for ny in range(max(0, y - 1), min(h, y + 2)):
                for nx in range(max(0, x - 1), min(w, x + 2)):
                    if not visited[ny, nx] and binary[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if area >= min_area:
            components.append({"area": area, "x0": x0, "y0": y0, "x1": x1 + 1, "y1": y1 + 1})
    components.sort(key=lambda row: int(row.get("area") or 0), reverse=True)
    return components


def _border_center_ratio(binary: "np.ndarray", border_frac: float = 0.18) -> Tuple[float, float]:
    if np is None or binary.size == 0:
        return 0.0, 0.0
    h, w = binary.shape[:2]
    by = max(1, int(round(h * border_frac)))
    bx = max(1, int(round(w * border_frac)))
    border = np.zeros_like(binary, dtype=bool)
    border[:by, :] = True
    border[-by:, :] = True
    border[:, :bx] = True
    border[:, -bx:] = True
    center = ~border
    border_ratio = float(np.mean(binary[border])) if np.any(border) else 0.0
    center_ratio = float(np.mean(binary[center])) if np.any(center) else 0.0
    return border_ratio, center_ratio


def _largest_contour_stats(mask: "np.ndarray") -> Dict[str, float]:
    if cv2 is None or np is None or mask.size == 0:
        return {"area_ratio": 0.0, "circularity": 0.0, "extent": 0.0}
    binary = (np.asarray(mask, dtype=np.uint8) * 255)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area_ratio": 0.0, "circularity": 0.0, "extent": 0.0}
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 1.0:
        return {"area_ratio": 0.0, "circularity": 0.0, "extent": 0.0}
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = float(max(1, w * h))
    full_area = float(max(1, mask.shape[0] * mask.shape[1]))
    circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 1.0 else 0.0
    return {
        "area_ratio": area / full_area,
        "circularity": float(circularity),
        "extent": area / bbox_area,
    }


def _component_layout_stats(comps: List[Dict[str, int]], width: int, height: int) -> Dict[str, float]:
    if not comps or width <= 0 or height <= 0:
        return {
            "span_x": 0.0,
            "span_y": 0.0,
            "weighted_cx": 0.0,
            "weighted_cy": 0.0,
            "edge_share": 0.0,
        }
    total_area = float(sum(max(1, int(comp.get("area") or 0)) for comp in comps))
    min_x = min(int(comp.get("x0") or 0) for comp in comps)
    min_y = min(int(comp.get("y0") or 0) for comp in comps)
    max_x = max(int(comp.get("x1") or 0) for comp in comps)
    max_y = max(int(comp.get("y1") or 0) for comp in comps)
    weighted_cx = 0.0
    weighted_cy = 0.0
    edge_area = 0.0
    edge_limit_x = max(1, int(round(width * 0.10)))
    edge_limit_y = max(1, int(round(height * 0.12)))
    for comp in comps:
        area = float(max(1, int(comp.get("area") or 0)))
        x0 = int(comp.get("x0") or 0)
        y0 = int(comp.get("y0") or 0)
        x1 = int(comp.get("x1") or 0)
        y1 = int(comp.get("y1") or 0)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        weighted_cx += cx * area
        weighted_cy += cy * area
        if x0 <= edge_limit_x or x1 >= (width - edge_limit_x) or y0 <= edge_limit_y or y1 >= (height - edge_limit_y):
            edge_area += area
    return {
        "span_x": (max_x - min_x) / float(width),
        "span_y": (max_y - min_y) / float(height),
        "weighted_cx": (weighted_cx / total_area) / float(width),
        "weighted_cy": (weighted_cy / total_area) / float(height),
        "edge_share": edge_area / total_area,
    }


def _gradient_axis_energy(crop_gray: "np.ndarray") -> Tuple[float, float, float]:
    if cv2 is None or np is None or crop_gray.size == 0:
        return 0.0, 0.0, 0.0
    gray32 = crop_gray.astype(np.float32)
    gx = cv2.Sobel(gray32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray32, cv2.CV_32F, 0, 1, ksize=3)
    abs_gx = np.abs(gx)
    abs_gy = np.abs(gy)
    diag = np.abs(gx - gy) * 0.5
    scale = 255.0 * float(max(1, crop_gray.shape[0] * crop_gray.shape[1]))
    return (
        float(np.sum(abs_gx)) / scale,
        float(np.sum(abs_gy)) / scale,
        float(np.sum(diag)) / scale,
    )


def _score_signature(crop_rgb: "np.ndarray", crop_gray: "np.ndarray") -> float:
    if np is None:
        return 0.0
    h, w = crop_gray.shape[:2]
    if h < 8 or w < 16:
        return 0.0
    aspect = float(w) / float(max(1, h))
    if aspect < 2.4 or aspect > 9.5:
        return 0.0
    dark = crop_gray < min(180, int(np.percentile(crop_gray, 42)))
    ink_ratio = float(np.mean(dark))
    row_t, col_t = _transition_density(dark)
    contrast = float(np.std(crop_gray)) / 255.0
    row_profile = np.mean(dark, axis=1)
    col_profile = np.mean(dark, axis=0)
    row_bands = _active_segments(row_profile, 0.02, min_len=max(2, h // 18), max_gap=max(1, h // 28))
    col_bands = _active_segments(col_profile, 0.03, min_len=max(3, w // 18), max_gap=max(2, w // 30))
    active_rows = float(np.mean(row_profile > 0.015))
    active_cols = float(np.mean(col_profile > 0.015))
    dense_rows = float(np.mean(row_profile > 0.22))
    dense_cols = float(np.mean(col_profile > 0.28))
    min_area = max(8, int(round((h * w) * 0.0025)))
    comps = _mask_components(dark, min_area=min_area)
    contour_stats = _largest_contour_stats(dark)
    layout_stats = _component_layout_stats(comps, w, h)
    comp_count = len(comps)
    total_dark = max(1, int(np.count_nonzero(dark)))
    largest_ratio = (int(comps[0]["area"]) / float(total_dark)) if comps else 0.0
    tiny_share = (
        sum(1 for comp in comps if int(comp.get("area") or 0) <= max(min_area * 2, 20)) / float(max(1, comp_count))
        if comps else 1.0
    )
    line_penalty = max(float(np.max(row_profile, initial=0.0)), float(np.max(col_profile, initial=0.0)))
    border_ratio, center_ratio = _border_center_ratio(dark, border_frac=0.10)
    gx_energy, gy_energy, diag_energy = _gradient_axis_energy(crop_gray)
    edge_touch_count = sum(
        1
        for edge_ratio in (
            float(np.mean(dark[0, :])),
            float(np.mean(dark[-1, :])),
            float(np.mean(dark[:, 0])),
            float(np.mean(dark[:, -1])),
        )
        if edge_ratio > 0.05
    )
    score = 0.0
    score += min(0.18, max(0.0, (aspect - 2.2) * 0.055))
    if 0.010 <= ink_ratio <= 0.16:
        score += 0.15
    elif 0.006 <= ink_ratio <= 0.22:
        score += 0.08
    if 1 <= len(row_bands) <= 2:
        score += 0.12
    if 1 <= len(col_bands) <= 3:
        score += 0.08
    if 1 <= comp_count <= 12:
        score += 0.16
    elif comp_count <= 18:
        score += 0.06
    if 0.08 <= largest_ratio <= 0.72:
        score += 0.11
    elif largest_ratio > 0.82:
        score -= 0.12
    if 0.05 <= contour_stats["extent"] <= 0.48:
        score += 0.06
    elif contour_stats["extent"] > 0.62:
        score -= 0.16
    if 0.18 <= active_rows <= 0.82:
        score += 0.08
    if 0.20 <= active_cols <= 0.92:
        score += 0.06
    if 0.34 <= layout_stats["span_x"] <= 0.94:
        score += 0.10
    else:
        score -= 0.12
    if 0.10 <= layout_stats["span_y"] <= 0.64:
        score += 0.06
    else:
        score -= 0.08
    if 0.16 <= layout_stats["weighted_cx"] <= 0.88:
        score += 0.04
    if 0.24 <= layout_stats["weighted_cy"] <= 0.78:
        score += 0.05
    score += min(0.10, row_t * 0.38)
    score += min(0.07, contrast * 0.20)
    if 0.15 <= diag_energy <= 1.10:
        score += 0.06
    if gx_energy > (gy_energy * 1.8):
        score -= 0.06
    score -= dense_rows * 0.20
    score -= dense_cols * 0.20
    score -= max(0.0, line_penalty - 0.42) * 0.35
    score -= max(0.0, border_ratio - center_ratio) * 0.45
    score -= max(0, edge_touch_count - 1) * 0.16
    score -= max(0, len(row_bands) - 2) * 0.12
    score -= max(0, comp_count - 18) * 0.02
    score -= tiny_share * 0.10
    score -= max(0.0, layout_stats["edge_share"] - 0.38) * 0.28
    score -= max(0.0, col_t - 0.26) * 0.18
    if contour_stats["circularity"] > 0.45:
        score -= 0.10
    if comp_count <= 1 and largest_ratio > 0.74:
        score -= 0.16
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
    color_mask = red_mask | blue_mask
    color_ratio = float(np.mean(color_mask))
    dark = crop_gray < min(180, int(np.percentile(crop_gray, 45)))
    dark_ratio = float(np.mean(dark))
    contour_stats = _largest_contour_stats(color_mask if color_ratio >= 0.01 else dark)
    aspect = float(w) / float(max(1, h))
    if not (0.72 <= aspect <= 1.35):
        return 0.0
    square_bonus = max(0.0, 1.0 - abs(aspect - 1.0) * 2.2)
    border_ratio, center_ratio = _border_center_ratio(color_mask if color_ratio >= 0.01 else dark)
    ring_hint = max(0.0, border_ratio - center_ratio)
    contrast = float(np.std(crop_gray)) / 255.0
    gx_energy, gy_energy, _ = _gradient_axis_energy(crop_gray)
    if color_ratio < 0.018 and ring_hint < 0.10:
        return 0.0
    score = 0.0
    score += min(0.46, color_ratio * 4.5)
    score += min(0.20, ring_hint * 0.75)
    score += square_bonus * 0.14
    if 0.18 <= contour_stats["circularity"] <= 1.15:
        score += 0.10
    if 0.10 <= contour_stats["extent"] <= 0.70:
        score += 0.05
    score += min(0.08, dark_ratio * 0.18)
    score += min(0.05, contrast * 0.12)
    if 0.10 <= gx_energy <= 1.5 and 0.10 <= gy_energy <= 1.5:
        score += 0.04
    if center_ratio > 0.48:
        score -= (center_ratio - 0.48) * 0.28
    if contour_stats["circularity"] < 0.08 and ring_hint < 0.16:
        score -= 0.12
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
    if not (0.16 <= black_ratio <= 0.62):
        return 0.0
    corner = max(3, int(round(min(h, w) * 0.22)))
    corners = [
        binary[:corner, :corner],
        binary[:corner, -corner:],
        binary[-corner:, :corner],
        binary[-corner:, -corner:],
    ]
    corner_scores = sorted((float(np.mean(part)) for part in corners if part.size), reverse=True)
    finder_hint = float(sum(corner_scores[:3]) / 3.0) if len(corner_scores) >= 3 else 0.0
    balance = max(0.0, 1.0 - abs(black_ratio - 0.38) * 2.4)
    gx_energy, gy_energy, _ = _gradient_axis_energy(crop_gray)
    axis_balance = max(0.0, 1.0 - abs(gx_energy - gy_energy) / float(max(0.15, gx_energy + gy_energy)))
    score = (row_t * 0.34) + (col_t * 0.34) + (balance * 0.18) + (finder_hint * 0.24)
    score += axis_balance * 0.08
    if finder_hint < 0.18:
        score -= 0.16
    if row_t < 0.18 or col_t < 0.18:
        score -= 0.10
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
    if not (0.15 <= black_ratio <= 0.72):
        return 0.0
    col_profile = np.mean(binary, axis=0)
    row_profile = np.mean(binary, axis=1)
    stripe_var = float(np.std(col_profile))
    stripe_jump = float(np.mean(np.abs(np.diff(col_profile)) > 0.10)) if col_profile.size > 1 else 0.0
    row_var = float(np.std(row_profile))
    gx_energy, gy_energy, _ = _gradient_axis_energy(crop_gray)
    aspect_bonus = min(0.28, (aspect - 1.8) * 0.06)
    balance = max(0.0, 1.0 - abs(black_ratio - 0.42) * 2.1)
    score = (
        (row_t * 0.36)
        + min(0.20, stripe_var * 0.80)
        + min(0.16, stripe_jump * 0.30)
        + (balance * 0.14)
        + aspect_bonus
        - (col_t * 0.12)
        - (row_var * 0.12)
    )
    if gx_energy > (gy_energy * 1.35):
        score += 0.08
    else:
        score -= 0.10
    if stripe_var < 0.10 or stripe_jump < 0.08:
        score -= 0.12
    return max(0.0, min(score, 1.0))


def _detection_geom(row: Dict[str, Any]) -> Dict[str, float]:
    bbox = row.get("bbox_norm") or {}
    x0 = max(0.0, min(1.0, _safe_float(bbox.get("x0"), 0.0)))
    y0 = max(0.0, min(1.0, _safe_float(bbox.get("y0"), 0.0)))
    x1 = max(x0, min(1.0, _safe_float(bbox.get("x1"), x0)))
    y1 = max(y0, min(1.0, _safe_float(bbox.get("y1"), y0)))
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    area = w * h
    cx = x0 + (w / 2.0)
    cy = y0 + (h / 2.0)
    aspect = (w / h) if h > 0.0 else 0.0
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "w": w, "h": h, "area": area, "cx": cx, "cy": cy, "aspect": aspect}


def _passes_type_geometry(row: Dict[str, Any]) -> bool:
    kind = str(row.get("type") or "")
    g = _detection_geom(row)
    if kind == "signature":
        return (
            0.08 <= g["w"] <= 0.68
            and 0.015 <= g["h"] <= 0.22
            and 2.4 <= g["aspect"] <= 11.5
            and 0.48 <= g["cy"] <= 0.93
            and g["area"] <= 0.08
        )
    if kind == "stamp":
        return (
            0.04 <= g["w"] <= 0.32
            and 0.04 <= g["h"] <= 0.32
            and 0.68 <= g["aspect"] <= 1.45
            and g["area"] <= 0.09
        )
    if kind == "qrcode":
        return (
            0.03 <= g["w"] <= 0.28
            and 0.03 <= g["h"] <= 0.28
            and 0.74 <= g["aspect"] <= 1.28
            and g["area"] <= 0.08
        )
    if kind == "barcode":
        return (
            0.06 <= g["w"] <= 0.70
            and 0.012 <= g["h"] <= 0.18
            and 1.7 <= g["aspect"] <= 18.0
            and g["area"] <= 0.10
        )
    return True


def _prefer_decoder_detections(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    decoder_rows = [row for row in rows if str(row.get("source") or "") in {"pyzbar", "opencv-qrcode"}]
    if not decoder_rows:
        return rows
    kept: List[Dict[str, Any]] = []
    for row in rows:
        source = str(row.get("source") or "")
        kind = str(row.get("type") or "")
        if source in {"pyzbar", "opencv-qrcode"}:
            kept.append(row)
            continue
        if kind not in {"barcode", "qrcode"}:
            kept.append(row)
            continue
        bbox = row.get("bbox_px") or {}
        a = (
            _safe_int(bbox.get("x0")),
            _safe_int(bbox.get("y0")),
            _safe_int(bbox.get("x1")),
            _safe_int(bbox.get("y1")),
        )
        overlap = False
        for dec in decoder_rows:
            if str(dec.get("type") or "") != kind:
                continue
            db = dec.get("bbox_px") or {}
            b = (
                _safe_int(db.get("x0")),
                _safe_int(db.get("y0")),
                _safe_int(db.get("x1")),
                _safe_int(db.get("y1")),
            )
            if _safe_int(dec.get("page_index"), 0) == _safe_int(row.get("page_index"), 0) and _bbox_iou(a, b) > 0.18:
                overlap = True
                break
        if not overlap:
            kept.append(row)
    return kept


def _postfilter_page_detections(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        if not _passes_type_geometry(row):
            continue
        if str(row.get("type") or "") == "signature":
            score = _safe_float(row.get("score"), 0.0)
            source = str(row.get("source") or "")
            if source == "vision-heuristics" and score < 0.74:
                continue
        filtered.append(row)
    filtered = _prefer_decoder_detections(filtered)
    filtered.sort(key=lambda x: (str(x.get("type") or ""), -_safe_float(x.get("score"), 0.0)))
    return filtered


def _make_detection(
    det_type: str,
    score: float,
    width: int,
    height: int,
    page_index: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    source: str,
    *,
    decoded_type: str | None = None,
    decoded_value: str | None = None,
    variant: str | None = None,
) -> Dict[str, Any]:
    x0 = max(0, min(width, int(x0)))
    y0 = max(0, min(height, int(y0)))
    x1 = max(x0 + 1, min(width, int(x1)))
    y1 = max(y0 + 1, min(height, int(y1)))
    row: Dict[str, Any] = {
        "type": det_type,
        "score": round(float(score), 6),
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
        "source": source,
    }
    if decoded_type:
        row["decoded_type"] = decoded_type
    if decoded_value:
        row["decoded_value"] = decoded_value
    if variant:
        row["variant"] = variant
    return row


def _variant_images_for_decoder(img: Image.Image) -> List[Tuple[str, Image.Image]]:
    variants: List[Tuple[str, Image.Image]] = [("gray", img.convert("L"))]
    if cv2 is None or np is None:
        return variants
    try:
        gray = np.asarray(img.convert("L"), dtype=np.uint8)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("otsu", Image.fromarray(otsu)))
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        variants.append(("adaptive", Image.fromarray(adaptive)))
        up = cv2.resize(gray, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
        variants.append(("gray-x17", Image.fromarray(up)))
        _, up_otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("otsu-x17", Image.fromarray(up_otsu)))
    except Exception:
        pass
    return variants


def _dedupe_decoder_detections(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for row in sorted(rows, key=lambda x: float(x.get("score") or 0.0), reverse=True):
        bbox = row.get("bbox_px") or {}
        a = (
            _safe_int(bbox.get("x0")),
            _safe_int(bbox.get("y0")),
            _safe_int(bbox.get("x1")),
            _safe_int(bbox.get("y1")),
        )
        skip = False
        for other in kept:
            if str(other.get("type") or "") != str(row.get("type") or ""):
                continue
            if _safe_int(other.get("page_index"), 0) != _safe_int(row.get("page_index"), 0):
                continue
            ob = other.get("bbox_px") or {}
            b = (
                _safe_int(ob.get("x0")),
                _safe_int(ob.get("y0")),
                _safe_int(ob.get("x1")),
                _safe_int(ob.get("y1")),
            )
            same_value = str(other.get("decoded_value") or "") == str(row.get("decoded_value") or "")
            if _bbox_iou(a, b) > 0.30 or (same_value and same_value != ""):
                skip = True
                break
        if not skip:
            kept.append(row)
    return kept


def _detect_qr_barcode_decoders(img: Image.Image, page_index: int) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    width, height = img.size

    if zbar_decode is not None:
        for variant_name, variant_img in _variant_images_for_decoder(img):
            try:
                variant_w, variant_h = variant_img.size
                scale_x = float(width) / float(max(1, variant_w))
                scale_y = float(height) / float(max(1, variant_h))
                for sym in zbar_decode(variant_img):
                    rect = getattr(sym, "rect", None)
                    if rect is None:
                        continue
                    raw_x0 = _safe_int(getattr(rect, "left", 0))
                    raw_y0 = _safe_int(getattr(rect, "top", 0))
                    raw_w = max(1, _safe_int(getattr(rect, "width", 0)))
                    raw_h = max(1, _safe_int(getattr(rect, "height", 0)))
                    x0 = int(round(raw_x0 * scale_x))
                    y0 = int(round(raw_y0 * scale_y))
                    x1 = int(round((raw_x0 + raw_w) * scale_x))
                    y1 = int(round((raw_y0 + raw_h) * scale_y))
                    raw_type = str(getattr(sym, "type", "") or "").strip().upper()
                    det_type = "qrcode" if "QRCODE" in raw_type or "QR" == raw_type else "barcode"
                    decoded_value = getattr(sym, "data", b"").decode("utf-8", errors="ignore") or None
                    score = 0.995 if variant_name == "gray" else 0.985
                    detections.append(
                        _make_detection(
                            det_type,
                            score,
                            width,
                            height,
                            page_index,
                            x0,
                            y0,
                            x1,
                            y1,
                            "pyzbar",
                            decoded_type=raw_type or None,
                            decoded_value=decoded_value,
                            variant=variant_name,
                        )
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
                            _make_detection(
                                "qrcode",
                                0.98,
                                width,
                                height,
                                page_index,
                                x0,
                                y0,
                                x1,
                                y1,
                                "opencv-qrcode",
                                decoded_value=decoded_infos[idx] if idx < len(decoded_infos) else None,
                            )
                        )
        except Exception:
            pass

        try:
            barcode_detector = None
            if hasattr(cv2, "barcode_BarcodeDetector"):
                barcode_detector = cv2.barcode_BarcodeDetector()
            elif hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
                barcode_detector = cv2.barcode.BarcodeDetector()
            if barcode_detector is not None:
                gray = np.asarray(img.convert("L"), dtype=np.uint8)
                ok, decoded_infos, decoded_types, corners = barcode_detector.detectAndDecode(gray)
                if ok and corners is not None:
                    decoded_infos = list(decoded_infos or [])
                    decoded_types = list(decoded_types or [])
                    for idx, quad in enumerate(corners):
                        if quad is None:
                            continue
                        xs = [float(pt[0]) for pt in quad]
                        ys = [float(pt[1]) for pt in quad]
                        x0 = max(0, int(round(min(xs))))
                        y0 = max(0, int(round(min(ys))))
                        x1 = min(width, int(round(max(xs))))
                        y1 = min(height, int(round(max(ys))))
                        decoded_type = str(decoded_types[idx]) if idx < len(decoded_types) else None
                        decoded_value = str(decoded_infos[idx]) if idx < len(decoded_infos) else None
                        detections.append(
                            _make_detection(
                                "barcode",
                                0.985,
                                width,
                                height,
                                page_index,
                                x0,
                                y0,
                                x1,
                                y1,
                                "opencv-barcode",
                                decoded_type=decoded_type,
                                decoded_value=decoded_value,
                            )
                        )
        except Exception:
            pass

    return _dedupe_decoder_detections(detections)


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
        sizes = [(0.40, 0.10), (0.32, 0.08), (0.48, 0.12)]
        y_min, y_max = 0.52, 0.92
        scorer = _score_signature
        threshold = 0.72
    elif kind == "stamp":
        sizes = [(0.18, 0.18), (0.22, 0.16), (0.14, 0.14)]
        y_min, y_max = 0.10, 0.95
        scorer = _score_stamp
        threshold = 0.63
    elif kind == "qrcode":
        sizes = [(0.14, 0.14), (0.18, 0.18), (0.24, 0.24)]
        y_min, y_max = 0.05, 0.95
        scorer = _score_qrcode
        threshold = 0.72
    else:
        sizes = [(0.34, 0.10), (0.28, 0.08), (0.42, 0.12)]
        y_min, y_max = 0.05, 0.95
        scorer = _score_barcode
        threshold = 0.74

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
                center_x = (x + (win_w / 2.0)) / float(max(1, width))
                center_y = (y + (win_h / 2.0)) / float(max(1, height))
                if kind == "signature":
                    if not (0.10 <= center_x <= 0.92):
                        score -= 0.12
                    if 0.62 <= center_y <= 0.86:
                        score += 0.05
                    if center_y > 0.88:
                        score -= 0.18
                    if center_y < 0.54:
                        score -= 0.10
                    if x <= step_x or (x + win_w) >= (width - step_x):
                        score -= 0.08
                    if center_x < 0.12 or center_x > 0.94:
                        score -= 0.08
                elif kind == "stamp":
                    if 0.16 <= center_y <= 0.88:
                        score += 0.03
                    if center_y < 0.06 or center_y > 0.96:
                        score -= 0.12
                elif kind == "barcode":
                    if aspect := (float(win_w) / float(max(1, win_h))):
                        if aspect > 2.8:
                            score += 0.04
                    if center_y < 0.08 or center_y > 0.94:
                        score -= 0.08
                elif kind == "qrcode":
                    if center_y < 0.04 or center_y > 0.96 or center_x < 0.04 or center_x > 0.96:
                        score -= 0.10
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
        max_keep = SIGNATURE_MAX_PER_PAGE if kind == "signature" else 2
        if len(kept) >= max_keep:
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
    detections = _postfilter_page_detections(detections)
    detections.sort(key=lambda x: (str(x.get("type") or ""), -_safe_float(x.get("score"), 0.0)))
    return detections


def _build_page_previews(
    pages: List[Image.Image],
    sampled_pages: List[int],
    detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for row in detections:
        page_idx = _safe_int(row.get("page_index"), 0)
        if page_idx <= 0:
            continue
        by_page.setdefault(page_idx, []).append(row)

    previews: List[Dict[str, Any]] = []
    for page_number, img in zip(sampled_pages, pages):
        page_rows = by_page.get(int(page_number)) or []
        if not page_rows:
            continue
        data_url = _image_to_data_url(img)
        if not data_url:
            continue
        width, height = img.size
        previews.append(
            {
                "page_index": int(page_number),
                "page_width": int(width),
                "page_height": int(height),
                "image_data_url": data_url,
                "detections": page_rows,
            }
        )
        if len(previews) >= 6:
            break
    return previews


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
    seen_doc_keys: set[str] = set()

    for idx, doc in _iter_docs(ctx):
        doc_id = doc.get("doc_id")
        filename = str(doc.get("filename") or f"doc_{idx}")
        doc_key = _doc_key(doc_id, filename, idx)
        if doc_key in seen_doc_keys:
            continue
        seen_doc_keys.add(doc_key)
        source_path = _resolve_source_path(doc, source_map)
        pages, sampled_pages, total_pages = _load_doc_pages(
            source_path,
            max_pages=_safe_int(ctx.get("VISUAL_SCAN_MAX_PAGES"), MAX_SCAN_PAGES),
        )

        detections: List[Dict[str, Any]] = []
        for page_number, img in zip(sampled_pages, pages):
            detections.extend(_detect_page_marks(img, page_index=page_number))
        detections = _dedupe_doc_detections(detections)
        page_previews = _build_page_previews(pages, sampled_pages, detections)

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
                "doc_key": doc_key,
                "engine": "visual-100ml-hybrid-heuristics-v2-precision",
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
                "page_previews": page_previews,
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
