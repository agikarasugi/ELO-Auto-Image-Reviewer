import csv
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rich.table import Table

from . import console as cm
from .elo import EloRatings

THUMB_W = 400
THUMB_H = 300
PADDING = 20
HEADER_H = 60
LABEL_H = 70


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_csv(ratings: EloRatings, output_dir: Path) -> Path:
    ts = _timestamp()
    path = output_dir / f"elo_results_{ts}.csv"
    ranked = ratings.ranked()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "wins", "losses", "elo_score"])
        for stats in ranked:
            writer.writerow([stats.filename, stats.wins, stats.losses, f"{stats.elo:.2f}"])
    return path


def _load_thumbnail(image_path: Path) -> Image.Image | None:
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
        return img
    except Exception:
        return None


def _make_placeholder(text: str = "Could not load") -> Image.Image:
    img = Image.new("RGB", (THUMB_W, THUMB_H), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (THUMB_W - (bbox[2] - bbox[0])) // 2
    y = (THUMB_H - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, fill=(100, 100, 100), font=font)
    return img


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def write_top3_image(
    ratings: EloRatings,
    images_dir: Path,
    output_dir: Path,
) -> Path:
    ranked = ratings.ranked()
    top = ranked[: min(3, len(ranked))]
    n = len(top)

    col_w = THUMB_W + PADDING * 2
    total_w = col_w * n
    total_h = HEADER_H + THUMB_H + LABEL_H + PADDING

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header_font = _get_font(22)
    rank_font = _get_font(18)
    label_font = _get_font(14)

    # Header
    header_text = "ELO Auto Image Reviewer — Top 3"
    bbox = draw.textbbox((0, 0), header_text, font=header_font)
    hx = (total_w - (bbox[2] - bbox[0])) // 2
    draw.text((hx, 16), header_text, fill=(30, 30, 30), font=header_font)

    rank_labels = ["#1", "#2", "#3"]
    medal_colors = [(212, 175, 55), (160, 160, 160), (176, 141, 87)]

    for i, stats in enumerate(top):
        x_offset = i * col_w + PADDING

        # Thumbnail
        thumb = _load_thumbnail(images_dir / stats.filename)
        if thumb is None:
            thumb = _make_placeholder()

        # Center thumbnail horizontally in column
        thumb_x = x_offset + (THUMB_W - thumb.width) // 2
        thumb_y = HEADER_H
        canvas.paste(thumb, (thumb_x, thumb_y))

        # Rank label
        rank_text = rank_labels[i]
        ry = HEADER_H + THUMB_H + 8
        rbbox = draw.textbbox((0, 0), rank_text, font=rank_font)
        rx = x_offset + (THUMB_W - (rbbox[2] - rbbox[0])) // 2
        draw.text((rx, ry), rank_text, fill=medal_colors[i], font=rank_font)

        # ELO score
        elo_text = f"ELO: {stats.elo:.0f}"
        ey = ry + (rbbox[3] - rbbox[1]) + 4
        ebbox = draw.textbbox((0, 0), elo_text, font=label_font)
        ex = x_offset + (THUMB_W - (ebbox[2] - ebbox[0])) // 2
        draw.text((ex, ey), elo_text, fill=(60, 60, 60), font=label_font)

        # Filename (truncated)
        fname = stats.filename if len(stats.filename) <= 32 else stats.filename[:29] + "..."
        fy = ey + (ebbox[3] - ebbox[1]) + 4
        fbbox = draw.textbbox((0, 0), fname, font=label_font)
        fx = x_offset + (THUMB_W - (fbbox[2] - fbbox[0])) // 2
        draw.text((fx, fy), fname, fill=(100, 100, 100), font=label_font)

    ts = _timestamp()
    out_path = output_dir / f"top3_{ts}.png"
    canvas.save(str(out_path), "PNG")
    return out_path


def print_summary_table(ratings: EloRatings) -> None:
    table = Table(title="ELO Rankings", show_header=True, header_style="bold cyan")
    table.add_column("Rank", style="bold", justify="right", width=6)
    table.add_column("Filename", no_wrap=False)
    table.add_column("ELO", justify="right")
    table.add_column("Wins", justify="right")
    table.add_column("Losses", justify="right")
    table.add_column("Win %", justify="right")

    ranked = ratings.ranked()
    medal = {0: "[bold yellow]#1[/]", 1: "[bold white]#2[/]", 2: "[bold yellow]#3[/]"}

    for i, stats in enumerate(ranked):
        rank_str = medal.get(i, str(i + 1))
        win_pct = f"{stats.win_rate:.1f}%"
        table.add_row(
            rank_str,
            stats.filename,
            f"{stats.elo:.1f}",
            str(stats.wins),
            str(stats.losses),
            win_pct,
        )

    cm.console.print(table)
    if cm.file_console is not None:
        cm.file_console.print(table)
