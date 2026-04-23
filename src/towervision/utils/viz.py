"""Simple visualization helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def draw_boxes(
    image_path: Path,
    boxes: Sequence[tuple[float, float, float, float]],
    *,
    output_path: Path,
    color: str = "red",
) -> Path:
    """Draw bounding boxes over an image and save the result."""

    with Image.open(image_path) as image:
        canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for x, y, width, height in boxes:
        draw.rectangle((x, y, x + width, y + height), outline=color, width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def draw_labeled_boxes(
    image_path: Path,
    annotations: Sequence[tuple[str, tuple[float, float, float, float]]],
    *,
    output_path: Path,
    color_by_label: Mapping[str, str] | None = None,
) -> Path:
    """Draw labeled bounding boxes over an image and save the result."""

    palette = dict(color_by_label or {"torre": "red", "isoladores": "cyan"})
    with Image.open(image_path) as image:
        canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for label, bbox in annotations:
        x, y, width, height = bbox
        x0 = math.floor(x)
        y0 = math.floor(y)
        x1 = math.ceil(x + width)
        y1 = math.ceil(y + height)
        color = palette.get(label, "yellow")
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        text_anchor_y = max(0, y0 - 12)
        draw.text((x0, text_anchor_y), label, fill=color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def draw_mask_overlay(
    image_path: Path,
    mask_path: Path,
    *,
    output_path: Path,
    color: tuple[int, int, int] = (255, 64, 64),
    alpha: int = 120,
    outline_color: tuple[int, int, int] = (255, 255, 0),
) -> Path:
    """Overlay a binary mask on top of an image and save the result."""

    with Image.open(image_path) as image:
        base_image = image.convert("RGBA")
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")

    if mask.size != base_image.size:
        raise ValueError(
            f"Mask size {mask.size} does not match image size {base_image.size}"
        )

    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    overlay_pixels = overlay.load()
    mask_pixels = mask.load()
    width, height = base_image.size
    for y in range(height):
        for x in range(width):
            if mask_pixels[x, y] > 0:
                overlay_pixels[x, y] = (*color, alpha)

    contour = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    contour_draw = ImageDraw.Draw(contour)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if mask_pixels[x, y] == 0:
                continue
            if (
                mask_pixels[x - 1, y] == 0
                or mask_pixels[x + 1, y] == 0
                or mask_pixels[x, y - 1] == 0
                or mask_pixels[x, y + 1] == 0
            ):
                contour_draw.point((x, y), fill=outline_color + (255,))

    canvas = Image.alpha_composite(base_image, overlay)
    canvas = Image.alpha_composite(canvas, contour)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)
    return output_path


def draw_anomaly_heatmap_overlay(
    image_path: Path,
    anomaly_map: np.ndarray,
    *,
    output_path: Path,
    value_range: tuple[float, float] | None = None,
    alpha: int = 176,
    mask_path: Path | None = None,
    outline_color: tuple[int, int, int] = (64, 255, 64),
) -> Path:
    """Overlay a continuous anomaly heatmap on top of an image and save the result."""

    with Image.open(image_path) as image:
        base_image = image.convert("RGBA")

    if anomaly_map.ndim != 2:
        raise ValueError(f"Expected 2D anomaly map, got shape {anomaly_map.shape}")

    low, high = value_range or (float(np.min(anomaly_map)), float(np.max(anomaly_map)))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        normalized = np.zeros_like(anomaly_map, dtype=np.float32)
    else:
        normalized = np.clip((anomaly_map.astype(np.float32) - low) / (high - low), 0.0, 1.0)

    heatmap = Image.fromarray((normalized * 255.0).astype(np.uint8), mode="L").resize(
        base_image.size,
        resample=Image.Resampling.BILINEAR,
    )
    heatmap_array = np.asarray(heatmap, dtype=np.float32) / 255.0
    rgba_overlay = _heatmap_to_rgba(heatmap_array, alpha=alpha)
    overlay = Image.fromarray(rgba_overlay, mode="RGBA")

    canvas = Image.alpha_composite(base_image, overlay)
    if mask_path is not None and mask_path.exists():
        canvas = _draw_mask_outline(canvas, mask_path=mask_path, outline_color=outline_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)
    return output_path


def render_contact_sheet(
    items: Sequence[tuple[Path, str]],
    *,
    output_path: Path,
    columns: int = 4,
    thumbnail_size: tuple[int, int] = (280, 280),
    cell_padding: int = 16,
    title: str | None = None,
) -> Path:
    """Render a labeled contact sheet from a list of images."""

    if not items:
        raise ValueError("At least one image is required to render a contact sheet")

    font = ImageFont.load_default()
    text_line_height = 14
    label_height = text_line_height * 3
    title_height = 32 if title else 0
    rows = math.ceil(len(items) / columns)
    cell_width = thumbnail_size[0] + (cell_padding * 2)
    cell_height = thumbnail_size[1] + label_height + (cell_padding * 2)
    sheet_width = columns * cell_width
    sheet_height = rows * cell_height + title_height

    canvas = Image.new("RGB", (sheet_width, sheet_height), color="white")
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((cell_padding, 8), title, fill="black", font=font)

    for index, (image_path, label) in enumerate(items):
        row_index = index // columns
        column_index = index % columns
        x0 = column_index * cell_width + cell_padding
        y0 = row_index * cell_height + cell_padding + title_height

        with Image.open(image_path) as image:
            preview = ImageOps.contain(image.convert("RGB"), thumbnail_size)
        offset_x = x0 + (thumbnail_size[0] - preview.width) // 2
        offset_y = y0 + (thumbnail_size[1] - preview.height) // 2
        canvas.paste(preview, (offset_x, offset_y))

        label_lines = label.split("\n")
        text_y = y0 + thumbnail_size[1] + 6
        for line in label_lines[:3]:
            draw.text((x0, text_y), line, fill="black", font=font)
            text_y += text_line_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _heatmap_to_rgba(values: np.ndarray, *, alpha: int) -> np.ndarray:
    """Convert normalized heat values in [0, 1] into an RGBA heatmap."""

    clipped = np.clip(values.astype(np.float32), 0.0, 1.0)
    red = np.clip(1.8 * clipped - 0.4, 0.0, 1.0)
    green = np.clip(1.8 - 3.2 * np.abs(clipped - 0.5), 0.0, 1.0)
    blue = np.clip(1.6 * (1.0 - clipped) - 0.2, 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    rgba = np.zeros((*clipped.shape, 4), dtype=np.uint8)
    rgba[..., :3] = (rgb * 255.0).astype(np.uint8)
    rgba[..., 3] = (clipped * float(alpha)).astype(np.uint8)
    return rgba


def _draw_mask_outline(
    image: Image.Image,
    *,
    mask_path: Path,
    outline_color: tuple[int, int, int],
) -> Image.Image:
    """Draw the contour of a binary mask on top of an RGBA image."""

    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, resample=Image.Resampling.NEAREST)

    contour = Image.new("RGBA", image.size, (0, 0, 0, 0))
    contour_draw = ImageDraw.Draw(contour)
    mask_pixels = mask.load()
    width, height = image.size
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if mask_pixels[x, y] == 0:
                continue
            if (
                mask_pixels[x - 1, y] == 0
                or mask_pixels[x + 1, y] == 0
                or mask_pixels[x, y - 1] == 0
                or mask_pixels[x, y + 1] == 0
            ):
                contour_draw.point((x, y), fill=outline_color + (255,))
    return Image.alpha_composite(image, contour)
