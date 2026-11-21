"""Stickers: carregamento (PNG com canal alfa), animação por spritesheet e overlay."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from imageio_utils import read_image

# Convenção: "nome_f6.png" = spritesheet horizontal com 6 quadros.
_ANIM_RE = re.compile(r"^(?P<name>.+)_f(?P<n>\d+)$")


@dataclass
class Sticker:
    name: str
    frames: List[np.ndarray]  # lista de BGRA uint8
    fps: float = 8.0

    @property
    def animated(self) -> bool:
        return len(self.frames) > 1

    def frame_at(self, t: float) -> np.ndarray:
        """Quadro a exibir no instante t (segundos). Estáticos retornam sempre o 1º."""
        if not self.animated:
            return self.frames[0]
        return self.frames[int(t * self.fps) % len(self.frames)]


@dataclass
class PlacedSticker:
    sticker: Sticker
    x: int
    y: int
    scale: float  # largura do sticker como fração da largura da imagem


def _split_spritesheet(sheet: np.ndarray, n: int) -> List[np.ndarray]:
    fw = sheet.shape[1] // n
    return [np.ascontiguousarray(sheet[:, i * fw:(i + 1) * fw]) for i in range(n)]


def load_sticker(path: Path) -> Sticker:
    img = read_image(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError("Não foi possível ler o sticker: %s" % path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:  # sem alfa: assume totalmente opaco
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    stem = re.sub(r"^\d+_", "", path.stem)  # "1_estrela" -> "estrela" (prefixo só ordena)
    m = _ANIM_RE.match(stem)
    if m and int(m.group("n")) > 1:
        return Sticker(m.group("name"), _split_spritesheet(img, int(m.group("n"))))
    return Sticker(stem, [img])


def load_stickers(folder: Path) -> List[Sticker]:
    """Carrega todos os PNG da pasta (ordem alfabética); ignora arquivos inválidos."""
    out: List[Sticker] = []
    for p in sorted(Path(folder).glob("*.png")):
        try:
            out.append(load_sticker(p))
        except IOError as exc:
            print("[aviso] %s" % exc)
    return out


def scale_to_width(sprite: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    h, w = sprite.shape[:2]
    height = max(1, int(round(h * width / w)))
    interp = cv2.INTER_AREA if width < w else cv2.INTER_LINEAR
    return cv2.resize(sprite, (width, height), interpolation=interp)


def overlay_rgba(base: np.ndarray, sprite: np.ndarray, cx: int, cy: int) -> None:
    """Desenha `sprite` (BGRA) em `base` (BGR) centrado em (cx, cy), com blending
    por alfa. Opera *in-place* e recorta o que sair da imagem."""
    sh, sw = sprite.shape[:2]
    x0 = int(round(cx - sw / 2.0))
    y0 = int(round(cy - sh / 2.0))
    bx0, by0 = max(x0, 0), max(y0, 0)
    bx1, by1 = min(x0 + sw, base.shape[1]), min(y0 + sh, base.shape[0])
    if bx0 >= bx1 or by0 >= by1:
        return
    sx0, sy0 = bx0 - x0, by0 - y0
    part = sprite[sy0:sy0 + (by1 - by0), sx0:sx0 + (bx1 - bx0)]
    alpha = part[:, :, 3:4].astype(np.float32) / 255.0
    roi = base[by0:by1, bx0:bx1].astype(np.float32)
    mixed = part[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha)
    base[by0:by1, bx0:bx1] = np.clip(mixed + 0.5, 0, 255).astype(np.uint8)


def draw_sticker(base: np.ndarray, placed: PlacedSticker, t: float) -> None:
    sprite = scale_to_width(placed.sticker.frame_at(t), placed.scale * base.shape[1])
    overlay_rgba(base, sprite, placed.x, placed.y)
