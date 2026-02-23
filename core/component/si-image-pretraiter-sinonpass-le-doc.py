from __future__ import annotations

import csv
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Sequence, Union, List

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # In notebooks __file__ is undefined; fall back to current working directory.
    SCRIPT_DIR = Path.cwd()

DEFAULT_LANG = "fra"
DEFAULT_CONTRAST = 1.5
DEFAULT_SHARPNESS = 1.2
DEFAULT_BRIGHTNESS = 1.0
DEFAULT_UPSCALE = 1.5
DEFAULT_DPI = 300

# Heuristiques
MIN_CHARS_OFFICE = 1
MIN_CHARS_PDF = 30
PDF_MAX_PAGES = 3
SEARCH_DIRS = [os.getcwd(), "/mnt/data"]  # utile en notebook


@dataclass(frozen=True)
class FileType:
    ext: str
    mime: str
    label: str


def _read_head(path: str, n: int = 16384) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def normalize_input_files(x: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if "," not in s: 
            return [s]
        parts = next(csv.reader([s], skipinitialspace=True))
        return [p.strip() for p in parts if p.strip()]
    return [str(p).strip() for p in x if str(p).strip()]


def resolve_path(p: str) -> Optional[str]:
    p = os.path.expandvars(os.path.expanduser(p.strip()))
    if os.path.exists(p):
        return os.path.abspath(p)

    base = os.path.basename(p)
    for d in SEARCH_DIRS:
        alt = os.path.join(d, base)
        if os.path.exists(alt):
            return os.path.abspath(alt)

    return None


def detect_path_type(path: str) -> FileType:
    head = _read_head(path)

    if head.startswith(b"%PDF-"):
        return FileType(".pdf", "application/pdf", "PDF document")

    if head.startswith(b"II*\x00") or head.startswith(b"MM\x00*"):
        return FileType(".tif", "image/tiff", "TIFF image")

    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return FileType(".png", "image/png", "PNG image")

    if head.startswith(b"\xff\xd8\xff"):
        return FileType(".jpg", "image/jpeg", "JPEG image")

    if len(head) >= 12 and head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return FileType(".webp", "image/webp", "WEBP image")

    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06") or head.startswith(b"PK\x07\x08"):
        try:
            with zipfile.ZipFile(path, "r") as z:
                names = set(z.namelist())

                if "mimetype" in names and "META-INF/container.xml" in names:
                    try:
                        mt = z.read("mimetype")[:64].decode("ascii", errors="ignore").strip()
                    except Exception:
                        mt = ""
                    if mt == "application/epub+zip":
                        return FileType(".epub", "application/epub+zip", "EPUB eBook")

                if "word/document.xml" in names:
                    return FileType(".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "Word document (DOCX)")
                if "xl/workbook.xml" in names:
                    return FileType(".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "Excel workbook (XLSX)")
                if "ppt/presentation.xml" in names:
                    return FileType(".pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation", "PowerPoint presentation (PPTX)")

                if "content.xml" in names and "META-INF/manifest.xml" in names:
                    mt = ""
                    try:
                        if "mimetype" in names:
                            mt = z.read("mimetype")[:128].decode("ascii", errors="ignore").strip()
                    except Exception:
                        mt = ""
                    if mt == "application/vnd.oasis.opendocument.text":
                        return FileType(".odt", mt, "OpenDocument Text (ODT)")
                    if mt == "application/vnd.oasis.opendocument.spreadsheet":
                        return FileType(".ods", mt, "OpenDocument Spreadsheet (ODS)")
                    if mt == "application/vnd.oasis.opendocument.presentation":
                        return FileType(".odp", mt, "OpenDocument Presentation (ODP)")
                    return FileType(".odf", "application/zip", "OpenDocument container")
        except Exception:
            pass

        return FileType(".zip", "application/zip", "ZIP archive/container")

    if head.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
        return FileType(".ole", "application/x-ole-storage", "OLE2 container (old Office)")

    return FileType("", "application/octet-stream", "Unknown / binary")


def _xml_text_len(xml_bytes: bytes) -> int:
    try:
        root = ET.fromstring(xml_bytes)
        total = 0
        for elem in root.iter():
            if elem.text and elem.text.strip():
                total += len(elem.text.strip())
        return total
    except Exception:
        s = re.sub(rb"<[^>]+>", b" ", xml_bytes)
        return len(re.sub(rb"\s+", b" ", s).strip())


def _zip_has_text(path: str, ext: str) -> bool:
    try:
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()

            if ext == ".docx":
                total = 0
                if "word/document.xml" in names:
                    total += _xml_text_len(z.read("word/document.xml"))
                for nm in names:
                    if nm.startswith("word/header") and nm.endswith(".xml"):
                        total += _xml_text_len(z.read(nm))
                    if nm.startswith("word/footer") and nm.endswith(".xml"):
                        total += _xml_text_len(z.read(nm))
                    if total >= MIN_CHARS_OFFICE:
                        break
                return total >= MIN_CHARS_OFFICE

            if ext == ".xlsx":
                total = 0
                if "xl/sharedStrings.xml" in names:
                    total += _xml_text_len(z.read("xl/sharedStrings.xml"))
                if total < MIN_CHARS_OFFICE:
                    for nm in names:
                        if nm.startswith("xl/worksheets/") and nm.endswith(".xml"):
                            total += _xml_text_len(z.read(nm))
                            if total >= MIN_CHARS_OFFICE:
                                break
                return total >= MIN_CHARS_OFFICE

            if ext == ".pptx":
                total = 0
                for nm in names:
                    if nm.startswith("ppt/slides/") and nm.endswith(".xml"):
                        total += _xml_text_len(z.read(nm))
                        if total >= MIN_CHARS_OFFICE:
                            break
                return total >= MIN_CHARS_OFFICE

            if ext in {".odt", ".ods", ".odp"}:
                return ("content.xml" in names) and (_xml_text_len(z.read("content.xml")) >= MIN_CHARS_OFFICE)

            if ext == ".epub":
                total = 0
                for nm in names:
                    low = nm.lower()
                    if low.endswith((".xhtml", ".html", ".htm")):
                        try:
                            b = z.read(nm)
                        except Exception:
                            continue
                        s = re.sub(rb"<[^>]+>", b" ", b)
                        total += len(re.sub(rb"\s+", b" ", s).strip())
                        if total >= MIN_CHARS_OFFICE:
                            break
                return total >= MIN_CHARS_OFFICE

    except Exception:
        return False

    return False


def _get_pdf_reader():
    try:
        from pypdf import PdfReader  # type: ignore
        return PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            return PdfReader
        except ImportError:
            return None


def _pdf_has_text(path: str) -> bool:
    PdfReader = _get_pdf_reader()

    if PdfReader is None:
        try:
            with open(path, "rb") as f:
                data = f.read(2_000_000)
            if b"/Font" in data:
                return True
            if b"BT" in data and (b"Tj" in data or b"TJ" in data):
                return True
        except Exception:
            pass
        return False

    try:
        reader = PdfReader(path)
        pages = reader.pages[: max(1, PDF_MAX_PAGES)]
        extracted_score = 0

        for page in pages:
            txt = page.extract_text() or ""
            extracted_score += len("".join(txt.split()))
            if extracted_score >= MIN_CHARS_PDF:
                return True
        return False
    except Exception:
        return False


def content_kind_two_states(path: str, ftype: FileType) -> str:
    ext = ftype.ext.lower()

    if ext in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".ico"}:
        return "image_only"

    if ext == ".pdf":
        return "text" if _pdf_has_text(path) else "image_only"

    if ext in {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".epub"}:
        return "text" if _zip_has_text(path, ext) else "image_only"

    return "image_only"


# --------- ROUTAGE ---------
ORIGINAL_INPUT_FILE = globals().get("INPUT_FILE", None)
_raw_items = normalize_input_files(ORIGINAL_INPUT_FILE)

IMAGE_ONLY_FILES: List[str] = []
TEXT_FILES: List[str] = []
MISSING_FILES: List[str] = []

for item in _raw_items:
    p = resolve_path(item)
    if p is None:
        MISSING_FILES.append(item)
        continue

    ft = detect_path_type(p)
    kind = content_kind_two_states(p, ft)

    if kind == "image_only":
        IMAGE_ONLY_FILES.append(p)
    else:
        TEXT_FILES.append(p)
        print(f"[skip] content='text' -> {p}")

# IMPORTANT: ton code OCR (cellule suivante) reste inchangé, il lira INPUT_FILE ici
INPUT_FILE = IMAGE_ONLY_FILES

if MISSING_FILES:
    print("[missing] fichiers introuvables:")
    for m in MISSING_FILES:
        print(" -", m)








SHOW_PREPROCESSED = True   #/////////////////////////////////////////////////////////////////////////////////////////////////////////////////


@dataclass
class EnhanceOptions:
    contrast: float = DEFAULT_CONTRAST
    sharpness: float = DEFAULT_SHARPNESS
    brightness: float = DEFAULT_BRIGHTNESS
    upscale: float = DEFAULT_UPSCALE
    gamma: Optional[float] = None  # gamma correction; <1 brightens darks, >1 darkens
    pad: int = 0  # pixels to pad around the image
    median: Optional[int] = None  # kernel size for median filter (odd int, e.g., 3)
    unsharp_radius: Optional[float] = None  # e.g., 1.0
    unsharp_percent: int = 150
    invert: bool = False
    autocontrast_cutoff: Optional[int] = None  # 0-100; percentage to clip for autocontrast
    equalize: bool = False  # histogram equalization
    auto_rotate: bool = False  # attempt orientation detection + rotate
    otsu: bool = False  # auto-threshold with Otsu (requires numpy)
    threshold: Optional[int] = None  # 0-255; if set, applies a binary threshold


def build_config(
    oem: Optional[int],
    psm: Optional[int],
    base_flags: Iterable[str],
    dpi: Optional[int],
    tessdata_dir: Optional[Path],
    user_words: Optional[Path],
    user_patterns: Optional[Path],
) -> str:
    parts: List[str] = []
    if oem is not None:
        parts.append(f"--oem {oem}")
    if psm is not None:
        parts.append(f"--psm {psm}")
    if dpi is not None:
        parts.append(f"--dpi {dpi}")
    if tessdata_dir is not None:
        parts.append(f'--tessdata-dir "{tessdata_dir}"')
    if user_words is not None:
        parts.append(f'--user-words "{user_words}"')
    if user_patterns is not None:
        parts.append(f'--user-patterns "{user_patterns}"')
    parts.extend(base_flags)
    return " ".join(parts)


def ensure_environment(lang: str) -> None:
    try:
        _ = pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        sys.exit("Tesseract binary not found on PATH. Install it and its language data.")
    if lang:
        try:
            available = set(pytesseract.get_languages(config=""))
            requested = set(lang.split("+"))
            missing = requested - available
            if missing:
                print(
                    f"Warning: missing languages: {', '.join(sorted(missing))}. "
                    f"Available: {', '.join(sorted(available))}",
                    file=sys.stderr,
                )
        except pytesseract.TesseractError:
            pass


def auto_rotate_if_needed(img: Image.Image, enhance: EnhanceOptions) -> Image.Image:
    if not enhance.auto_rotate:
        return img
    try:
        osd = pytesseract.image_to_osd(img)
        angle = None
        for line in osd.splitlines():
            if line.lower().startswith("rotate:"):
                try:
                    angle = int(line.split(":")[1].strip())
                except ValueError:
                    angle = None
                break
        if angle is not None and angle % 360 != 0:
            return img.rotate(-angle, expand=True)
    except Exception:
        pass
    return img


def preprocess_image(image: Image.Image, enhance: EnhanceOptions) -> Image.Image:
    img = image.convert("L")
    img = auto_rotate_if_needed(img, enhance)

    if enhance.invert:
        img = ImageOps.invert(img)

    if enhance.pad and enhance.pad > 0:
        img = ImageOps.expand(img, border=enhance.pad, fill=255)

    if enhance.autocontrast_cutoff is not None:
        cutoff = max(0, min(100, enhance.autocontrast_cutoff))
        img = ImageOps.autocontrast(img, cutoff=cutoff)

    if enhance.equalize:
        img = ImageOps.equalize(img)

    if enhance.upscale and enhance.upscale != 1.0:
        w, h = img.size
        img = img.resize((int(w * enhance.upscale), int(h * enhance.upscale)), Image.LANCZOS)

    if enhance.gamma and enhance.gamma > 0:
        inv_gamma = 1.0 / enhance.gamma
        lut = [pow(x / 255.0, inv_gamma) * 255 for x in range(256)]
        img = img.point(lut)

    if enhance.brightness and enhance.brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(enhance.brightness)

    if enhance.contrast and enhance.contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(enhance.contrast)

    if enhance.sharpness and enhance.sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(enhance.sharpness)

    if enhance.unsharp_radius:
        img = img.filter(
            ImageFilter.UnsharpMask(
                radius=enhance.unsharp_radius,
                percent=enhance.unsharp_percent,
                threshold=0,
            )
        )

    if enhance.median and enhance.median > 1 and enhance.median % 2 == 1:
        img = img.filter(ImageFilter.MedianFilter(size=enhance.median))

    if enhance.threshold is not None:
        thr = max(0, min(255, enhance.threshold))
        img = img.point(lambda p, t=thr: 255 if p > t else 0, mode="1").convert("L")
    elif enhance.otsu and np is not None:
        arr = np.array(img, dtype=np.uint8)
        hist, _ = np.histogram(arr, bins=256, range=(0, 256))
        total = arr.size
        sum_total = np.dot(np.arange(256), hist)

        sum_b = 0.0
        w_b = 0.0
        max_var = 0.0
        threshold = 0

        for i in range(256):
            w_b += hist[i]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += i * hist[i]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = w_b * w_f * (m_b - m_f) ** 2
            if var_between > max_var:
                max_var = var_between
                threshold = i

        img = img.point(lambda p, t=threshold: 255 if p > t else 0, mode="1").convert("L")

    return img


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lang", default=DEFAULT_LANG)
    parser.add_argument("--oem", type=int, choices=range(0, 4), default=None)
    parser.add_argument("--psm", type=int, choices=range(0, 14), default=None)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--tessdata-dir", type=Path, default=None)
    parser.add_argument("--user-words", type=Path, default=None)
    parser.add_argument("--user-patterns", type=Path, default=None)
    parser.add_argument("--whitelist", type=str, default=None)
    parser.add_argument("--blacklist", type=str, default=None)

    parser.add_argument("--contrast", type=float, default=DEFAULT_CONTRAST)
    parser.add_argument("--sharpness", type=float, default=DEFAULT_SHARPNESS)
    parser.add_argument("--brightness", type=float, default=DEFAULT_BRIGHTNESS)
    parser.add_argument("--upscale", type=float, default=DEFAULT_UPSCALE)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument("--median", type=int, default=None)
    parser.add_argument("--unsharp-radius", type=float, default=None)
    parser.add_argument("--unsharp-percent", type=int, default=150)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--autocontrast-cutoff", type=int, default=None)
    parser.add_argument("--equalize", action="store_true")
    parser.add_argument("--auto-rotate", action="store_true")
    parser.add_argument("--otsu", action="store_true")

    parser.add_argument(
        "--config",
        nargs="*",
        default=[],
        metavar="CFG",
        help="Additional configuration flags passed verbatim to tesseract (e.g., -c foo=bar).",
    )

    return parser.parse_args(list(argv) if argv is not None else [])


#  Exécution Cellule 1 (jusqu’à l’affichage) 

args = parse_args()
ensure_environment(args.lang)

enhance = EnhanceOptions(
    contrast=args.contrast,
    sharpness=args.sharpness,
    brightness=args.brightness,
    upscale=args.upscale,
    gamma=args.gamma,
    pad=args.pad,
    median=args.median,
    unsharp_radius=args.unsharp_radius,
    unsharp_percent=args.unsharp_percent,
    invert=args.invert,
    autocontrast_cutoff=args.autocontrast_cutoff,
    equalize=args.equalize,
    auto_rotate=args.auto_rotate,
    otsu=args.otsu,
    threshold=args.threshold,
)

config_flags: List[str] = list(args.config)

# AJOUTE ÇA :
config_flags.append("-c preserve_interword_spaces=1")

if args.whitelist:
    config_flags.append(f"-c tessedit_char_whitelist={args.whitelist}")
if args.blacklist:
    config_flags.append(f"-c tessedit_char_blacklist={args.blacklist}")


def _normalize_input_files(val):
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        items = list(val)
    else:
        items = [val]

    out = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, Path):
            out.append(str(item))
            continue
        s = str(item).strip()
        if not s:
            continue
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            out.extend(parts)
        else:
            out.append(s)
    return out

# Backwards-compatible alias (older cell name)
_normalize_input_file = _normalize_input_files

# Safeguard if INPUT_FILE cell not executed yet
INPUT_FILE = globals().get("INPUT_FILE", None)


def _load_images_from_path(path: Path, dpi: int):
    if path.suffix.lower() == ".pdf":
        try:
            from pdf2image import convert_from_path
        except Exception:
            sys.exit(
                "pdf2image is not available. Install it and Poppler to read PDF files."
            )
        try:
            return convert_from_path(str(path), dpi=dpi)
        except Exception as exc:
            sys.exit(f"PDF conversion failed for {path}: {exc}")
    # default: image file (supports multi-page TIFF)
    img = Image.open(path)
    n_frames = getattr(img, "n_frames", 1)
    if n_frames and n_frames > 1:
        images = []
        for i in range(n_frames):
            try:
                img.seek(i)
            except Exception:
                break
            images.append(img.copy())
        return images
    return [img]


input_items = _normalize_input_files(INPUT_FILE)
if not input_items:
    print("[info] Aucun fichier à OCR (image_only). Tout ce que tu as donné est détecté comme 'text'.")
    DOCS = []
else:
    DOCS = []
    for item in input_items:
        path = Path(item)
        if not path.is_absolute():
            path = (SCRIPT_DIR / path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"INPUT_FILE not found: {path}")

        print(f"[info] Using INPUT_FILE={path}", file=sys.stderr)

        dpi_val = int(getattr(args, "dpi", DEFAULT_DPI) or DEFAULT_DPI)
        images = _load_images_from_path(path, dpi=dpi_val)

        if len(images) == 1:
            original = images[0]
            prepped = preprocess_image(original, enhance)
            DOCS.append({"path": path, "original": original, "prepped": prepped})
        else:
            total = len(images)
            for idx, original in enumerate(images, start=1):
                prepped = preprocess_image(original, enhance)
                DOCS.append({
                    "path": path,
                    "original": original,
                    "prepped": prepped,
                    "page_index": idx,
                    "page_count": total
                })


DOCS = []
for item in input_items:
    path = Path(item)
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()

    if not path.exists():
        sys.exit(f"INPUT_FILE not found: {path}")

    print(f"[info] Using INPUT_FILE={path}", file=sys.stderr)

    dpi_val = int(getattr(args, "dpi", DEFAULT_DPI) or DEFAULT_DPI)
    images = _load_images_from_path(path, dpi=dpi_val)

    if len(images) == 1:
        original = images[0]
        prepped = preprocess_image(original, enhance)
        DOCS.append({"path": path, "original": original, "prepped": prepped})
    else:
        total = len(images)
        for idx, original in enumerate(images, start=1):
            prepped = preprocess_image(original, enhance)
            DOCS.append({
                "path": path,
                "original": original,
                "prepped": prepped,
                "page_index": idx,
                "page_count": total
            })

from IPython.display import display

for doc in DOCS:
    original = doc["original"]
    prepped = doc["prepped"]
    path = doc["path"]

    display(original.convert("RGB") if original.mode not in ("RGB","L") else original)

    if "SHOW_PREPROCESSED" not in globals() or SHOW_PREPROCESSED:
        display(prepped.convert("RGB") if prepped.mode not in ("RGB","L") else prepped)

# Keep globals aligned with the last document for backwards compatibility.
if DOCS:
    path = DOCS[-1]["path"]
    original = DOCS[-1]["original"]
    prepped = DOCS[-1]["prepped"]

