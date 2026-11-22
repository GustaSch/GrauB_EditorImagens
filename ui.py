"""Interface desenhada dentro da janela do OpenCV: barra de botões, carrossel de filtros com
miniaturas, sliders de parâmetro, paleta de stickers, barra de mensagem e painel de atalhos.

Tudo aqui é só desenho + registro de "áreas clicáveis" (hits) em `ed._hits`; quem reage ao
clique é `Editor.on_mouse`. As funções recebem o `Editor` (sem importá-lo: evita ciclo).
Cada hit é uma tupla (x0, y0, x1, y1, tipo, argumento, extra).
"""
from __future__ import annotations

import textwrap
import unicodedata
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from filters import apply_effect
from imageio_utils import read_image

# ------------------------------------------------------------------ medidas
TOOLBAR_H = 34
THUMB_W, THUMB_H = 96, 58
THUMB_BASE_W = 320          # largura da imagem-base usada para gerar as miniaturas
CAR_H = 84
ARROW_W = 22
SLIDER_ROWS = 3             # nº máximo de parâmetros de um efeito (Bilateral)
SLIDER_ROW_H = 20
SLIDERS_H = SLIDER_ROWS * SLIDER_ROW_H + 6
STK_H = 50
MSG_H = 24
LEGEND_W = 300
CHROME_H = TOOLBAR_H + CAR_H + SLIDERS_H + STK_H + MSG_H   # altura de tudo que fica sob o HUD

# ------------------------------------------------------------------- cores (BGR)
BG = (28, 28, 28)
PANEL = (44, 44, 44)
BTN = (66, 66, 66)
BTN_ON = (150, 105, 30)
ACCENT = (255, 190, 80)
TEXT = (235, 235, 235)
DIM = (150, 150, 150)
GOOD = (90, 200, 110)
WARN = (80, 200, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX

LEGEND = [
    ("l", "carregar foto"),
    ("i", "2a imagem (operacoes)"),
    ("v", "webcam liga/desliga"),
    ("n / b", "proximo / anterior efeito"),
    ("p", "proximo parametro"),
    ("+ / -", "ajusta o parametro"),
    ("m", "canal RGB>GRAY>R>G>B"),
    ("a", "aplicar (Enter / Espaco)"),
    ("c", "encadear filtros"),
    ("z", "desfazer"),
    ("r", "resetar tudo"),
    ("s", "salvar em output/"),
    ("1-9", "sticker (0 = nenhum)"),
    (", / .", "tamanho do sticker"),
    ("f", "filtro de gato (rosto)"),
    ("h", "mostrar/ocultar atalhos"),
    ("ESC / q", "sair"),
]
MOUSE_TIPS = ("Mouse: clique na miniatura para escolher o efeito (duplo clique aplica); "
              "arraste os sliders para ajustar; escolha um sticker e clique na foto para coloca-lo.")


def ascii_text(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def _clip(s: str, n: int) -> str:
    s = ascii_text(s)
    return s if len(s) <= n else s[:max(1, n - 2)] + ".."


def text_w(label: str, scale: float = 0.45) -> int:
    return cv2.getTextSize(label, FONT, scale, 1)[0][0]


def put(canvas, text, x, y, color=TEXT, scale=0.45) -> None:
    cv2.putText(canvas, ascii_text(text), (int(x), int(y)), FONT, scale, color, 1, cv2.LINE_AA)


def _rect(canvas, x0, y0, x1, y1, color, thickness=-1) -> None:
    cv2.rectangle(canvas, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), color, thickness)


# --------------------------------------------------------------- miniaturas
def _sample_image() -> np.ndarray:
    path = Path(__file__).resolve().parent / "assets" / "samples" / "exemplo1.jpg"
    img = read_image(path, cv2.IMREAD_COLOR) if path.exists() else None
    if img is None:
        img = np.zeros((200, 320, 3), np.uint8)
        img[:, :, 0] = np.linspace(0, 255, 320, dtype=np.uint8)[None, :]
        img[:, :, 2] = np.linspace(255, 0, 200, dtype=np.uint8)[:, None]
    return img


def _make_base(img: np.ndarray) -> np.ndarray:
    """Reduz para ~320 px de largura e recorta no formato da miniatura (96:58)."""
    h, w = img.shape[:2]
    scale = THUMB_BASE_W / float(w)
    small = cv2.resize(img, (THUMB_BASE_W, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    target_h = int(round(sw * THUMB_H / float(THUMB_W)))
    if sh >= target_h:
        y = (sh - target_h) // 2
        return np.ascontiguousarray(small[y:y + target_h])
    target_w = int(round(sh * THUMB_W / float(THUMB_H)))
    x = max(0, (sw - target_w) // 2)
    return np.ascontiguousarray(small[:, x:x + target_w])


def _thumb_base(ed) -> np.ndarray:
    if ed.mode == "photo":
        src, token = ed.original, ("photo", id(ed.original))
    elif ed.mode == "video" and ed._thumb_snapshot is not None:
        src, token = ed._thumb_snapshot, ("video", id(ed._thumb_snapshot))
    else:
        src, token = None, ("sample",)
    if ed._thumb_token != token:
        ed._thumb_token = token
        ed._thumb_cache.clear()
        ed._thumb_base = _make_base(src if src is not None else _sample_image())
    return ed._thumb_base


def thumbnail(ed, effect) -> np.ndarray:
    """Miniatura (THUMB_H x THUMB_W) do efeito aplicado à imagem atual com o canal e os
    parâmetros correntes. Com cache: só recalcula se a imagem, o canal ou os parâmetros mudam."""
    base = _thumb_base(ed)
    params = ed.param_values[effect.key]
    other_src = ed.ensure_second_image() if effect.kind == "operation" else None
    key = (effect.key, ed.channel, tuple(sorted(params.items())),
           id(other_src) if other_src is not None else None)
    th = ed._thumb_cache.get(key)
    if th is not None:
        return th
    try:
        other = None
        if other_src is not None:
            other = cv2.resize(other_src, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)
        out = apply_effect(base, effect, dict(params), ed.channel, other)
        th = cv2.resize(out, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    except Exception:  # miniatura nunca deve derrubar a interface
        th = np.full((THUMB_H, THUMB_W, 3), 70, np.uint8)
        put(th, "?", THUMB_W // 2 - 4, THUMB_H // 2 + 5, TEXT, 0.6)
    if len(ed._thumb_cache) > 400:
        ed._thumb_cache.clear()
    ed._thumb_cache[key] = th
    return th


def sticker_icon(ed, idx: int, size: int = 38) -> np.ndarray:
    icon = ed._sticker_icons.get(idx)
    if icon is not None:
        return icon
    frame = ed.stickers[idx].frames[0]
    h, w = frame.shape[:2]
    s = size / float(max(h, w))
    small = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    icon = np.full((size, size, 3), 80, np.uint8)
    oy, ox = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
    alpha = small[:, :, 3:4].astype(np.float32) / 255.0
    region = icon[oy:oy + small.shape[0], ox:ox + small.shape[1]].astype(np.float32)
    icon[oy:oy + small.shape[0], ox:ox + small.shape[1]] = (
        small[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)
    ed._sticker_icons[idx] = icon
    return icon


# ------------------------------------------------------------------ blocos
TOOLBAR_GAP = 4
TOOLBAR_PAD = 16


def toolbar_buttons(ed) -> List[Tuple[str, int, bool]]:
    return [
        ("Carregar", ord("l"), False),
        ("2a imagem", ord("i"), False),
        ("Webcam", ord("v"), ed.mode == "video"),
        ("Aplicar", ord("a"), False),
        ("Desfazer", ord("z"), False),
        ("Resetar", ord("r"), False),
        ("Salvar", ord("s"), False),
        ("Canal: " + ed.channel, ord("m"), ed.channel != "RGB"),
        ("Encadear", ord("c"), ed.chain),
        ("Gato (rosto)", ord("f"), ed.face_on),
        ("Atalhos", ord("h"), ed.show_legend),
    ]


def toolbar_width(ed) -> int:
    """Largura necessária para a barra de botões. A largura do texto do OpenCV varia entre
    versões/sistemas, então é medida em vez de fixada; usa o rótulo mais largo do canal
    ("Canal: GRAY") para a janela não mudar de tamanho ao trocar de canal."""
    labels = [b[0] for b in toolbar_buttons(ed)] + ["Canal: GRAY"]
    labels.remove("Canal: " + ed.channel)
    return 6 + sum(text_w(l) + TOOLBAR_PAD + TOOLBAR_GAP for l in labels) + 2


def draw_toolbar(ed, canvas, y: int, cw: int) -> None:
    _rect(canvas, 0, y, cw, y + TOOLBAR_H, PANEL)
    x = 6
    for label, key, active in toolbar_buttons(ed):
        w = text_w(label) + TOOLBAR_PAD
        _rect(canvas, x, y + 5, x + w, y + TOOLBAR_H - 5, BTN_ON if active else BTN)
        put(canvas, label, x + TOOLBAR_PAD // 2, y + 22, TEXT)
        ed._hits.append((x, y + 5, x + w, y + TOOLBAR_H - 5, "key", key, None))
        x += w + TOOLBAR_GAP


def draw_carousel(ed, canvas, y: int, cw: int) -> None:
    _rect(canvas, 0, y, cw, y + CAR_H, BG)
    n = len(ed.effects)
    cell = THUMB_W + 8
    vis = max(1, min(n, (cw - 2 * ARROW_W - 12) // cell))
    start = max(0, min(n - vis, ed.effect_idx - vis // 2))
    for sym, x0, delta in (("<", 2, -(vis - 1)), (">", cw - ARROW_W - 2, vis - 1)):
        _rect(canvas, x0, y + 4, x0 + ARROW_W, y + 4 + THUMB_H, BTN)
        put(canvas, sym, x0 + 6, y + 4 + THUMB_H // 2 + 5, TEXT, 0.6)
        ed._hits.append((x0, y + 4, x0 + ARROW_W, y + 4 + THUMB_H, "page", delta, None))
    applied = {s.effect.key for s in ed.steps}
    x = 2 + ARROW_W + 6
    for i in range(start, start + vis):
        e = ed.effects[i]
        canvas[y + 4:y + 4 + THUMB_H, x:x + THUMB_W] = thumbnail(ed, e)
        selected = i == ed.effect_idx
        _rect(canvas, x - 1, y + 3, x + THUMB_W + 1, y + 5 + THUMB_H,
              ACCENT if selected else (90, 90, 90), 2 if selected else 1)
        if e.key in applied:
            cv2.circle(canvas, (x + 9, y + 13), 4, GOOD, -1, cv2.LINE_AA)
        if e.kind == "operation":
            put(canvas, "A+B", x + THUMB_W - 26, y + 15, WARN, 0.38)
        put(canvas, _clip(e.name.split(" (")[0], 15), x, y + 4 + THUMB_H + 14,
            ACCENT if selected else TEXT, 0.36)
        ed._hits.append((x, y + 3, x + THUMB_W, y + 5 + THUMB_H + 16, "effect", i, None))
        x += cell


def _slider_track(cw: int) -> Tuple[int, int]:
    return 130, max(300, min(cw, 900) - 150)


def draw_sliders(ed, canvas, y: int, cw: int) -> None:
    _rect(canvas, 0, y, cw, y + SLIDERS_H, PANEL)
    params = ed.effect.params
    x0, x1 = _slider_track(cw)
    if not params:
        put(canvas, "(este efeito nao tem parametros)", 10, y + 18, DIM)
    for i in range(SLIDER_ROWS):
        if i >= len(params):
            continue
        p = params[i]
        ry = y + 3 + i * SLIDER_ROW_H
        mid = ry + SLIDER_ROW_H // 2
        sel = i == ed.param_idx
        put(canvas, _clip(p.name, 14), 10, mid + 5, ACCENT if sel else TEXT)
        cv2.line(canvas, (x0, mid), (x1, mid), (110, 110, 110), 3, cv2.LINE_AA)
        frac = (ed.params[p.name] - p.min) / float(p.max - p.min) if p.max > p.min else 0.0
        frac = max(0.0, min(1.0, frac))
        tx = int(x0 + frac * (x1 - x0))
        cv2.line(canvas, (x0, mid), (tx, mid), ACCENT if sel else (200, 200, 200), 3, cv2.LINE_AA)
        cv2.circle(canvas, (tx, mid), 6, ACCENT if sel else (200, 200, 200), -1, cv2.LINE_AA)
        put(canvas, "%s" % ed.params[p.name], x1 + 16, mid + 5, TEXT)
        put(canvas, "[%s .. %s]" % (p.min, p.max), x1 + 70, mid + 5, DIM, 0.36)
        ed._hits.append((0, ry, x0 - 10, ry + SLIDER_ROW_H, "param", i, None))
        ed._hits.append((x0 - 10, ry, x1 + 10, ry + SLIDER_ROW_H, "slider", i, (x0, x1)))


def draw_stickers(ed, canvas, y: int, cw: int) -> None:
    _rect(canvas, 0, y, cw, y + STK_H, BG)
    put(canvas, "Stickers:", 10, y + 30, TEXT)
    x = 84
    for i in range(len(ed.stickers)):
        icon = sticker_icon(ed, i)
        canvas[y + 6:y + 6 + icon.shape[0], x:x + icon.shape[1]] = icon
        sel = ed.sticker_sel == i
        _rect(canvas, x - 1, y + 5, x + icon.shape[1] + 1, y + 7 + icon.shape[0],
              ACCENT if sel else (90, 90, 90), 2 if sel else 1)
        put(canvas, str(i + 1), x + 2, y + 18, WARN, 0.4)
        ed._hits.append((x, y + 5, x + icon.shape[1], y + 7 + icon.shape[0], "sticker", i, None))
        x += icon.shape[1] + 6
    for label, kind, arg in (("nenhum", "sticker_none", None), ("-", "sticker_size", -1),
                              ("+", "sticker_size", 1)):
        w = text_w(label) + 16
        _rect(canvas, x + 6, y + 10, x + 6 + w, y + 36, BTN)
        put(canvas, label, x + 14, y + 28, TEXT)
        ed._hits.append((x + 6, y + 10, x + 6 + w, y + 36, kind, arg, None))
        x += w + 4
    put(canvas, "tamanho %.0f%%" % (ed.sticker_scale * 100), x + 12, y + 28, DIM)


def draw_message(ed, canvas, y: int, cw: int) -> None:
    _rect(canvas, 0, y, cw, y + MSG_H, (20, 20, 20))
    put(canvas, _clip(ed.message or "Pronto. Use o mouse ou os atalhos (h mostra/oculta a legenda).",
                      max(20, cw // 7)), 10, y + 17, (200, 235, 200), 0.45)


def draw_interface(ed, canvas, y: int, cw: int) -> None:
    """Desenha toolbar, carrossel, sliders, stickers e mensagem a partir da linha `y`."""
    draw_toolbar(ed, canvas, y, cw)
    y += TOOLBAR_H
    draw_carousel(ed, canvas, y, cw)
    y += CAR_H
    draw_sliders(ed, canvas, y, cw)
    y += SLIDERS_H
    draw_stickers(ed, canvas, y, cw)
    y += STK_H
    draw_message(ed, canvas, y, cw)


def draw_legend(canvas, x: int) -> None:
    """Painel lateral com os atalhos (da coluna x até a borda direita)."""
    h = canvas.shape[0]
    _rect(canvas, x, 0, x + LEGEND_W, h, (38, 38, 38))
    cv2.line(canvas, (x, 0), (x, h), (80, 80, 80), 1)
    put(canvas, "ATALHOS", x + 14, 26, ACCENT, 0.6)
    y = 54
    for key, text in LEGEND:
        put(canvas, key, x + 14, y, ACCENT, 0.45)
        put(canvas, text, x + 84, y, TEXT, 0.42)
        y += 21
    y += 8
    for line in textwrap.wrap(MOUSE_TIPS, 42):
        put(canvas, line, x + 14, y, DIM, 0.38)
        y += 16


def slider_value(p, x: float, x0: int, x1: int):
    """Valor do parâmetro `p` para a posição x do mouse no trilho [x0, x1] (alinhado ao passo)."""
    frac = 0.0 if x1 <= x0 else max(0.0, min(1.0, (x - x0) / float(x1 - x0)))
    v = p.min + frac * (p.max - p.min)
    if p.step > 0:
        v = p.min + round((v - p.min) / p.step) * p.step
    return p.clamp(v)
