"""Gera os stickers (PNG com canal alfa) e as imagens de exemplo do projeto.

Uso:  python generate_assets.py

Os stickers são desenhados proceduralmente com OpenCV (sem direitos autorais).
Arquivo "nome_f6.png" é um spritesheet horizontal de 6 quadros (sticker animado).
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from imageio_utils import write_image

ROOT = Path(__file__).resolve().parent
STICKERS_DIR = ROOT / "assets" / "stickers"
FACE_DIR = STICKERS_DIR / "face"
SAMPLES_DIR = ROOT / "assets" / "samples"

SS = 4  # supersampling para bordas suaves


def _canvas(w: int, h: int) -> np.ndarray:
    return np.zeros((h * SS, w * SS, 4), dtype=np.uint8)


def _finish(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _pt(x, y):
    return (int(round(x * SS)), int(round(y * SS)))


def _circle(img, c, r, color, thickness=-1):
    cv2.circle(img, _pt(*c), int(round(r * SS)), color, thickness if thickness < 0 else thickness * SS,
               cv2.LINE_AA)


def _poly(img, pts, color):
    cv2.fillPoly(img, [np.array([_pt(*p) for p in pts], np.int32)], color, cv2.LINE_AA)


def _star_points(cx, cy, r_out, r_in, n=5, rot=-math.pi / 2):
    pts = []
    for i in range(n * 2):
        r = r_out if i % 2 == 0 else r_in
        a = rot + i * math.pi / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def make_star(size=256):
    img = _canvas(size, size)
    c = size / 2
    _poly(img, _star_points(c, c * 1.06, size * 0.48, size * 0.20), (0, 140, 230, 255))
    _poly(img, _star_points(c, c * 1.06, size * 0.40, size * 0.17), (40, 210, 255, 255))
    return _finish(img, size, size)


def make_heart(size=256):
    img = _canvas(size, size)
    pts = []
    for t in np.linspace(0, 2 * math.pi, 200):
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((size / 2 + x * size / 36.0, size * 0.46 - y * size / 36.0))
    _poly(img, pts, (80, 60, 230, 255))
    _circle(img, (size * 0.33, size * 0.30), size * 0.07, (200, 200, 255, 255))
    return _finish(img, size, size)


def make_smiley(size=256):
    img = _canvas(size, size)
    c = size / 2
    _circle(img, (c, c), size * 0.47, (0, 140, 200, 255))
    _circle(img, (c, c), size * 0.43, (40, 220, 255, 255))
    _circle(img, (size * 0.35, size * 0.38), size * 0.06, (30, 30, 30, 255))
    _circle(img, (size * 0.65, size * 0.38), size * 0.06, (30, 30, 30, 255))
    cv2.ellipse(img, _pt(c, size * 0.55), (int(size * 0.22 * SS), int(size * 0.18 * SS)), 0, 10, 170,
                (30, 30, 30, 255), int(size * 0.035 * SS), cv2.LINE_AA)
    return _finish(img, size, size)


def make_sunglasses(w=320, h=130):
    img = _canvas(w, h)
    dark, rim = (35, 30, 30, 255), (15, 15, 15, 255)
    for cx in (w * 0.26, w * 0.74):
        cv2.ellipse(img, _pt(cx, h * 0.55), (int(w * 0.20 * SS), int(h * 0.38 * SS)), 0, 0, 360, rim, -1,
                    cv2.LINE_AA)
        cv2.ellipse(img, _pt(cx, h * 0.55), (int(w * 0.18 * SS), int(h * 0.32 * SS)), 0, 0, 360, dark, -1,
                    cv2.LINE_AA)
        cv2.line(img, _pt(cx - w * 0.10, h * 0.40), _pt(cx - w * 0.02, h * 0.30), (200, 200, 200, 255),
                 int(3 * SS), cv2.LINE_AA)
    cv2.line(img, _pt(w * 0.45, h * 0.45), _pt(w * 0.55, h * 0.45), rim, int(6 * SS), cv2.LINE_AA)
    cv2.line(img, _pt(w * 0.07, h * 0.40), _pt(2, h * 0.30), rim, int(6 * SS), cv2.LINE_AA)
    cv2.line(img, _pt(w * 0.93, h * 0.40), _pt(w - 2, h * 0.30), rim, int(6 * SS), cv2.LINE_AA)
    return _finish(img, w, h)


def make_crown(w=300, h=220):
    img = _canvas(w, h)
    gold, dark = (40, 200, 250, 255), (0, 130, 200, 255)
    pts = [(20, h - 20), (20, 70), (80, 130), (150, 30), (220, 130), (280, 70), (280, h - 20)]
    _poly(img, pts, dark)
    inner = [(34, h - 34), (34, 100), (84, 148), (150, 56), (216, 148), (266, 100), (266, h - 34)]
    _poly(img, inner, gold)
    for p, col in (((20, 70), (60, 60, 230, 255)), ((150, 30), (230, 120, 60, 255)),
                   ((280, 70), (60, 60, 230, 255))):
        _circle(img, p, 14, col)
    for x in (80, 150, 220):
        _circle(img, (x, h - 52), 11, (210, 240, 255, 255))
    return _finish(img, w, h)


def make_speech_bubble(w=320, h=240):
    img = _canvas(w, h)
    cv2.rectangle(img, _pt(14, 14), _pt(w - 14, h * 0.74), (60, 60, 60, 255), -1, cv2.LINE_AA)
    _poly(img, [(70, h * 0.70), (60, h - 12), (130, h * 0.70)], (60, 60, 60, 255))
    cv2.rectangle(img, _pt(24, 24), _pt(w - 24, h * 0.70), (255, 255, 255, 255), -1, cv2.LINE_AA)
    _poly(img, [(76, h * 0.68), (68, h - 26), (122, h * 0.68)], (255, 255, 255, 255))
    for i, x in enumerate((w * 0.30, w * 0.50, w * 0.70)):
        _circle(img, (x, h * 0.40), 14, (60, 60, 60, 255))
    return _finish(img, w, h)


def make_sparkle_sheet(frames=6, size=160):
    """Spritesheet horizontal: estrela de 4 pontas que pulsa e gira."""
    sheet = np.zeros((size, size * frames, 4), dtype=np.uint8)
    for i in range(frames):
        img = _canvas(size, size)
        phase = i / float(frames)
        k = 0.55 + 0.45 * abs(math.sin(math.pi * phase))
        rot = phase * math.pi / 2
        c = size / 2
        _poly(img, _star_points(c, c, size * 0.48 * k, size * 0.10 * k, n=4, rot=rot - math.pi / 2),
              (255, 230, 140, 255))
        _poly(img, _star_points(c, c, size * 0.30 * k, size * 0.07 * k, n=4, rot=rot - math.pi / 2 + math.pi / 4),
              (255, 255, 255, 255))
        sheet[:, i * size:(i + 1) * size] = _finish(img, size, size)
    return sheet


def make_cat_ears(w=400, h=200):
    img = _canvas(w, h)
    dark, pink = (60, 60, 70, 255), (190, 150, 255, 255)
    _poly(img, [(20, h - 6), (60, 10), (170, h - 6)], dark)
    _poly(img, [(380, h - 6), (340, 10), (230, h - 6)], dark)
    _poly(img, [(60, h - 20), (72, 50), (136, h - 20)], pink)
    _poly(img, [(340, h - 20), (328, 50), (264, h - 20)], pink)
    return _finish(img, w, h)


def make_cat_nose(w=300, h=160):
    img = _canvas(w, h)
    _poly(img, [(w * 0.42, h * 0.28), (w * 0.58, h * 0.28), (w * 0.50, h * 0.42)], (150, 120, 255, 255))
    cv2.line(img, _pt(w * 0.5, h * 0.42), _pt(w * 0.5, h * 0.52), (40, 40, 40, 255), int(3 * SS), cv2.LINE_AA)
    cv2.ellipse(img, _pt(w * 0.45, h * 0.52), (int(w * 0.05 * SS), int(h * 0.07 * SS)), 0, 0, 180,
                (40, 40, 40, 255), int(3 * SS), cv2.LINE_AA)
    cv2.ellipse(img, _pt(w * 0.55, h * 0.52), (int(w * 0.05 * SS), int(h * 0.07 * SS)), 0, 0, 180,
                (40, 40, 40, 255), int(3 * SS), cv2.LINE_AA)
    for sgn in (-1, 1):
        for dy in (-0.10, 0.04, 0.18):
            cv2.line(img, _pt(w * (0.5 + sgn * 0.12), h * (0.46 + dy * 0.5)),
                     _pt(w * (0.5 + sgn * 0.48), h * (0.40 + dy * 1.6)), (40, 40, 40, 255),
                     int(2 * SS), cv2.LINE_AA)
    return _finish(img, w, h)


def make_sample_scene(w=960, h=640, seed=1, variant=0):
    """Cena sintética colorida (céu, sol, colinas, casa) com ruído leve."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    top, bottom = ((200, 140, 60), (250, 220, 170)) if variant == 0 else ((60, 40, 120), (60, 150, 250))
    for y in range(h):
        t = y / float(h)
        img[y, :] = [int(top[i] * (1 - t) + bottom[i] * t) for i in range(3)]
    cv2.circle(img, (int(w * 0.78), int(h * 0.22)), int(h * 0.10), (120, 230, 255), -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(w * 0.25), int(h * 0.85)), (int(w * 0.55), int(h * 0.30)), 0, 180, 360,
                (70, 160, 70), -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(w * 0.80), int(h * 0.90)), (int(w * 0.50), int(h * 0.28)), 0, 180, 360,
                (50, 130, 60), -1, cv2.LINE_AA)
    hx, hy, hw, hh = int(w * 0.40), int(h * 0.52), int(w * 0.20), int(h * 0.22)
    cv2.rectangle(img, (hx, hy), (hx + hw, hy + hh), (150, 190, 235), -1)
    cv2.fillPoly(img, [np.array([(hx - 12, hy), (hx + hw // 2, hy - int(h * 0.12)), (hx + hw + 12, hy)])],
                 (40, 60, 160), cv2.LINE_AA)
    cv2.rectangle(img, (hx + hw // 2 - 18, hy + hh - 60), (hx + hw // 2 + 18, hy + hh), (40, 70, 110), -1)
    cv2.rectangle(img, (hx + 14, hy + 20), (hx + 54, hy + 56), (255, 240, 200), -1)
    cv2.rectangle(img, (hx + hw - 54, hy + 20), (hx + hw - 14, hy + 56), (255, 240, 200), -1)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def generate_all(verbose: bool = True) -> None:
    items = {
        STICKERS_DIR / "1_estrela.png": make_star(),
        STICKERS_DIR / "2_coracao.png": make_heart(),
        STICKERS_DIR / "3_sorriso.png": make_smiley(),
        STICKERS_DIR / "4_oculos.png": make_sunglasses(),
        STICKERS_DIR / "5_coroa.png": make_crown(),
        STICKERS_DIR / "6_balao.png": make_speech_bubble(),
        STICKERS_DIR / "7_brilho_f6.png": make_sparkle_sheet(),
        FACE_DIR / "orelhas.png": make_cat_ears(),
        FACE_DIR / "focinho.png": make_cat_nose(),
        SAMPLES_DIR / "exemplo1.jpg": make_sample_scene(variant=0, seed=1),
        SAMPLES_DIR / "exemplo2.jpg": make_sample_scene(variant=1, seed=2),
    }
    for path, img in items.items():
        if not write_image(path, img):
            raise IOError("Falha ao gravar %s" % path)
        if verbose:
            print("gerado:", path.relative_to(ROOT))


if __name__ == "__main__":
    generate_all()
