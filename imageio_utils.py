"""Leitura/escrita de imagens segura para caminhos com acentos (Windows).

`cv2.imread/imwrite` falham com caminhos não-ASCII em alguns sistemas; aqui
usamos numpy + imdecode/imencode para evitar o problema.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def read_image(path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_image(path, img: np.ndarray) -> bool:
    path = Path(path)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(path))
    except OSError:
        return False
    return True


def limit_size(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Reduz (nunca amplia) mantendo a proporção para caber em max_w x max_h."""
    h, w = img.shape[:2]
    s = min(max_w / float(w), max_h / float(h), 1.0)
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
