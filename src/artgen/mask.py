"""描き直す範囲の四角のマスクと、ぼかした貼り戻し。"""

from __future__ import annotations

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from artgen.config import ConfigError

Box = tuple[int, int, int, int]


def check_box(box: Box, size: tuple[int, int]) -> None:
    """四角が画像に収まり、x0<x1・y0<y1 でなければ止める。"""
    x0, y0, x1, y1 = box
    w, h = size
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ConfigError(
            f"inpaint_box: 画像 {w}x{h} の中に収まる x0<x1・y0<y1 の四角でない: {tuple(box)}"
        )


def make_mask(size: tuple[int, int], box: Box) -> Image.Image:
    """四角の中が 255、外が 0 のモード L のマスク（x1・y1 は含まない）。"""
    x0, y0, x1, y1 = box
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
    return mask


def feather_composite(
    original: Image.Image, generated: Image.Image, box: Box, feather: int
) -> Image.Image:
    """四角を内側に feather だけ縮めてぼかしたマスクで、generated を original に重ねる。

    四角の外は必ず original のまま。
    """
    if original.size != generated.size:
        raise ValueError(f"大きさが違う: {original.size} と {generated.size}")
    check_box(box, original.size)
    x0, y0, x1, y1 = box
    inner = (x0 + feather, y0 + feather, x1 - feather, y1 - feather)
    if inner[0] >= inner[2] or inner[1] >= inner[3]:
        raise ConfigError(f"inpaint feather: 四角 {tuple(box)} に対して大きすぎる: {feather}")
    mask = make_mask(original.size, inner)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather / 2))
        mask = ImageChops.multiply(mask, make_mask(original.size, box))
    return Image.composite(generated.convert("RGB"), original.convert("RGB"), mask)
