import base64
from pathlib import Path

from PIL import Image

from . import console as cm

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

MAX_ENCODE_PX = 2048


def collect_images(directory: Path) -> list[Path]:
    images = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and p.stat().st_size > 0
    )
    if not images:
        raise ValueError(f"No supported images found in {directory}")
    return images


def validate_image_count(
    images: list[Path],
    minimum: int = 5,
    warn_threshold: int = 10,
) -> None:
    n = len(images)
    unique_pairs = n * (n - 1) // 2
    if n < minimum:
        raise SystemExit(
            f"Error: found {n} image(s) in directory, but at least {minimum} are required "
            f"to compute a meaningful ELO ranking.\n"
            f"Add more images and try again."
        )
    if n < warn_threshold:
        cm.console.print(
            f"[yellow]Warning: only {n} images found ({unique_pairs} unique pairs). "
            f"ELO scores will be more reliable with 10+ images.[/yellow]"
        )


def _resize_for_api(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= MAX_ENCODE_PX:
        return img
    if w >= h:
        new_w = MAX_ENCODE_PX
        new_h = int(h * MAX_ENCODE_PX / w)
    else:
        new_h = MAX_ENCODE_PX
        new_w = int(w * MAX_ENCODE_PX / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def encode_image_base64(image_path: Path) -> tuple[str, str]:
    """Returns (base64_string, mime_type)."""
    mime = MIME_TYPES.get(image_path.suffix.lower(), "image/jpeg")
    with Image.open(image_path) as img:
        img = img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img
        img = _resize_for_api(img)
        import io
        buf = io.BytesIO()
        fmt = "PNG" if mime == "image/png" else "JPEG"
        img.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, mime


def build_image_content_block(image_path: Path) -> dict:
    """Returns an OpenAI image_url content block."""
    b64, mime = encode_image_base64(image_path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }
