"""Filtros e operações aritméticas do editor.

Cada efeito é descrito por um objeto `Effect` com nome, descrição textual,
parâmetros configuráveis e a função que o implementa.

Convenções (BGR / uint8):
  * Filtros:   func(img, **params)      -> img  (mesmo tamanho)
  * Operações: func(a, b, **params)     -> img  (a e b com o mesmo tamanho)
  * Todas as funções aceitam imagens de 1 canal (H, W) e de 3 canais (H, W, 3).
    Efeitos que só fazem sentido em cores declaram `color_only=True`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

# Modos de canal disponíveis (ordem do ciclo na interface).
CHANNELS = ["RGB", "GRAY", "R", "G", "B"]
_BGR_INDEX = {"B": 0, "G": 1, "R": 2}


# --------------------------------------------------------------------------
# Estruturas
# --------------------------------------------------------------------------
@dataclass
class Param:
    name: str
    label: str
    default: float
    min: float
    max: float
    step: float
    integer: bool = False

    def clamp(self, value: float):
        value = max(self.min, min(self.max, value))
        return int(round(value)) if self.integer else round(float(value), 3)


@dataclass
class Effect:
    key: str
    name: str
    kind: str  # "filter" | "operation"
    category: str
    description: str
    func: Callable
    params: List[Param] = field(default_factory=list)
    color_only: bool = False

    def default_params(self) -> Dict[str, float]:
        return {p.name: p.clamp(p.default) for p in self.params}


# --------------------------------------------------------------------------
# Auxiliares
# --------------------------------------------------------------------------
def _odd(k) -> int:
    k = max(1, int(round(k)))
    return k if k % 2 == 1 else k + 1


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _like(img: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Devolve `gray` no mesmo formato (1 ou 3 canais) de `img`."""
    return gray if img.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _to_u8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Filtros de suavização
# --------------------------------------------------------------------------
def box_blur(img, ksize=5):
    k = _odd(ksize)
    return cv2.blur(img, (k, k))


def gaussian_blur(img, ksize=7, sigma=0.0):
    k = _odd(ksize)
    return cv2.GaussianBlur(img, (k, k), float(sigma))


def median_blur(img, ksize=5):
    k = max(3, _odd(ksize))
    return cv2.medianBlur(img, k)


def bilateral(img, diameter=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, int(diameter), float(sigma_color), float(sigma_space))


# --------------------------------------------------------------------------
# Filtros de realce
# --------------------------------------------------------------------------
def sharpen(img, amount=1.0):
    a = float(amount)
    kernel = np.array([[0, -a, 0],
                       [-a, 1 + 4 * a, -a],
                       [0, -a, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def laplacian_enhance(img, strength=1.0):
    f = img.astype(np.float32)
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=3)
    return _to_u8(f - float(strength) * lap)


def unsharp_mask(img, ksize=7, amount=1.5):
    k = _odd(ksize)
    blur = cv2.GaussianBlur(img, (k, k), 0)
    return cv2.addWeighted(img, 1.0 + float(amount), blur, -float(amount), 0)


# --------------------------------------------------------------------------
# Bordas
# --------------------------------------------------------------------------
def sobel_edges(img, gain=1.0):
    g = _gray(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy) * float(gain)
    return _like(img, _to_u8(mag))


def canny_edges(img, low=80, high=160):
    g = _gray(img)
    lo, hi = sorted((float(low), float(high)))
    return _like(img, cv2.Canny(g, lo, hi))


def emboss(img):
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel, delta=128)


# --------------------------------------------------------------------------
# Cor / tons / histograma
# --------------------------------------------------------------------------
def grayscale(img):
    return _like(img, _gray(img))


def negative(img):
    return cv2.bitwise_not(img)


_SEPIA = np.array([[0.131, 0.534, 0.272],   # B' (entrada B, G, R)
                   [0.168, 0.686, 0.349],   # G'
                   [0.189, 0.769, 0.393]],  # R'
                  dtype=np.float32)


def sepia(img):
    return cv2.transform(img, _SEPIA)


def brightness_contrast(img, contrast=1.0, brightness=0):
    out = img.astype(np.float32) * float(contrast) + float(brightness)
    return _to_u8(out)


def gamma_correction(img, gamma=1.0):
    g = max(0.05, float(gamma))
    lut = (np.power(np.arange(256) / 255.0, 1.0 / g) * 255.0).astype(np.uint8)
    return cv2.LUT(img, lut)


def threshold(img, level=127):
    _, out = cv2.threshold(_gray(img), float(level), 255, cv2.THRESH_BINARY)
    return _like(img, out)


def equalize_hist(img):
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def pixelate(img, block=10):
    h, w = img.shape[:2]
    b = max(1, int(block))
    small = cv2.resize(img, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------
# Operações aritméticas entre duas imagens
# --------------------------------------------------------------------------
def op_add(a, b, weight=1.0):
    return cv2.add(a, cv2.convertScaleAbs(b, alpha=float(weight)))


def op_subtract(a, b, weight=1.0):
    return cv2.subtract(a, cv2.convertScaleAbs(b, alpha=float(weight)))


def op_blend(a, b, alpha=0.5):
    al = float(alpha)
    return cv2.addWeighted(a, 1.0 - al, b, al, 0)


def op_multiply(a, b):
    return cv2.multiply(a, b, scale=1.0 / 255.0)


def op_absdiff(a, b):
    return cv2.absdiff(a, b)


def op_lighten(a, b):
    return cv2.max(a, b)


# --------------------------------------------------------------------------
# Registro
# --------------------------------------------------------------------------
def _p(name, label, default, lo, hi, step, integer=False):
    return Param(name, label, default, lo, hi, step, integer)


EFFECTS: List[Effect] = [
    # ---- suavização
    Effect("box", "Box Blur", "filter", "suavizacao",
           "Suavização: substitui cada pixel pela média da vizinhança ksize x ksize. "
           "Reduz ruído e detalhes. Parâmetro: ksize (ímpar).",
           box_blur, [_p("ksize", "tamanho do kernel (ímpar)", 5, 3, 31, 2, True)]),
    Effect("gauss", "Gaussian Blur", "filter", "suavizacao",
           "Suavização: média ponderada por uma gaussiana (centro pesa mais que a borda). "
           "Parâmetros: ksize (ímpar) e sigma (0 = automático).",
           gaussian_blur, [_p("ksize", "tamanho do kernel (ímpar)", 7, 3, 31, 2, True),
                           _p("sigma", "desvio padrão (0 = auto)", 0, 0, 20, 0.5)]),
    Effect("median", "Median Blur", "filter", "suavizacao",
           "Suavização não-linear: usa a mediana da vizinhança. Remove ruído sal-e-pimenta "
           "preservando bordas. Parâmetro: ksize (ímpar).",
           median_blur, [_p("ksize", "tamanho do kernel (ímpar)", 5, 3, 31, 2, True)]),
    Effect("bilateral", "Bilateral", "filter", "suavizacao",
           "Suavização que preserva bordas: pondera vizinhos por distância espacial e por "
           "similaridade de cor. Parâmetros: diameter, sigma_color, sigma_space.",
           bilateral, [_p("diameter", "diâmetro da vizinhança", 9, 3, 15, 2, True),
                       _p("sigma_color", "sigma de cor", 75, 10, 200, 5),
                       _p("sigma_space", "sigma espacial", 75, 10, 200, 5)]),
    # ---- realce
    Effect("sharpen", "Sharpen", "filter", "realce",
           "Realce de nitidez por convolução com kernel [0,-a,0; -a,1+4a,-a; 0,-a,0]. "
           "Parâmetro: amount (intensidade a).",
           sharpen, [_p("amount", "intensidade", 1.0, 0.1, 3.0, 0.1)]),
    Effect("laplacian", "Laplaciano (realce)", "filter", "realce",
           "Realce pelo operador Laplaciano: saída = imagem - c * Laplaciano(imagem), "
           "acentuando transições de intensidade. Parâmetro: strength (c).",
           laplacian_enhance, [_p("strength", "intensidade", 1.0, 0.1, 3.0, 0.1)]),
    Effect("unsharp", "Unsharp Mask", "filter", "realce",
           "Realce: soma a imagem a diferença entre ela e sua versão borrada "
           "(img*(1+k) - blur*k). Parâmetros: ksize do blur e amount (k).",
           unsharp_mask, [_p("ksize", "tamanho do blur (ímpar)", 7, 3, 31, 2, True),
                          _p("amount", "intensidade", 1.5, 0.1, 5.0, 0.1)]),
    # ---- bordas
    Effect("sobel", "Sobel (bordas)", "filter", "bordas",
           "Detecção de bordas pelo módulo do gradiente de Sobel (horizontal + vertical). "
           "Parâmetro: gain (ganho do resultado).",
           sobel_edges, [_p("gain", "ganho", 1.0, 0.2, 5.0, 0.2)]),
    Effect("canny", "Canny (bordas)", "filter", "bordas",
           "Detecção de bordas de Canny (gradiente + supressão de não-máximos + histerese). "
           "Parâmetros: limiar baixo e alto.",
           canny_edges, [_p("low", "limiar baixo", 80, 0, 255, 10, True),
                         _p("high", "limiar alto", 160, 0, 255, 10, True)]),
    Effect("emboss", "Emboss (relevo)", "filter", "bordas",
           "Efeito de relevo por convolução direcional somada a um deslocamento de 128. "
           "Sem parâmetros.",
           emboss),
    # ---- cor / tons
    Effect("gray", "Grayscale", "filter", "cor",
           "Converte para tons de cinza (luminância: 0.299R + 0.587G + 0.114B). Sem parâmetros.",
           grayscale),
    Effect("negative", "Negativo", "filter", "cor",
           "Inverte as intensidades: saída = 255 - entrada. Sem parâmetros.",
           negative),
    Effect("sepia", "Sepia", "filter", "cor",
           "Tom envelhecido: transformação linear dos canais R,G,B por matriz de sepia. "
           "Sem parâmetros.",
           sepia, color_only=True),
    Effect("bc", "Brilho / Contraste", "filter", "cor",
           "Ajuste linear saída = contraste * entrada + brilho (com saturação em 0..255). "
           "Parâmetros: contrast e brightness.",
           brightness_contrast, [_p("contrast", "contraste (ganho)", 1.0, 0.2, 3.0, 0.1),
                                 _p("brightness", "brilho (soma)", 0, -100, 100, 10, True)]),
    Effect("gamma", "Gamma", "filter", "cor",
           "Correção gama por tabela (LUT): saída = 255*(entrada/255)^(1/gamma). "
           ">1 clareia, <1 escurece. Parâmetro: gamma.",
           gamma_correction, [_p("gamma", "gamma", 1.0, 0.2, 3.0, 0.1)]),
    Effect("threshold", "Limiarização", "filter", "cor",
           "Binariza a imagem: pixels acima do limiar viram branco, os demais preto. "
           "Parâmetro: level (0-255).",
           threshold, [_p("level", "limiar", 127, 0, 255, 8, True)]),
    Effect("equalize", "Equalização de histograma", "filter", "cor",
           "Redistribui as intensidades para aumentar o contraste global (em cores, "
           "equaliza apenas a luminância Y). Sem parâmetros.",
           equalize_hist),
    Effect("pixelate", "Pixelate", "filter", "cor",
           "Mosaico: reduz a imagem e amplia com vizinho mais próximo, formando blocos. "
           "Parâmetro: block (tamanho do bloco em pixels).",
           pixelate, [_p("block", "tamanho do bloco", 10, 2, 60, 2, True)]),
    # ---- operações com duas imagens
    Effect("op_add", "Adição (A + w*B)", "operation", "operacao",
           "Operação entre duas imagens: soma saturada A + w*B (clareia). "
           "Parâmetro: weight (peso da imagem B).",
           op_add, [_p("weight", "peso de B", 1.0, 0.0, 1.0, 0.1)]),
    Effect("op_sub", "Subtração ponderada (A - w*B)", "operation", "operacao",
           "Operação entre duas imagens: subtração saturada A - w*B (escurece/remove B). "
           "Parâmetro: weight (peso da imagem B).",
           op_subtract, [_p("weight", "peso de B", 1.0, 0.0, 2.0, 0.1)]),
    Effect("op_blend", "Blending (mistura)", "operation", "operacao",
           "Operação entre duas imagens: mistura (1-alpha)*A + alpha*B. "
           "Parâmetro: alpha (0 = só A, 1 = só B).",
           op_blend, [_p("alpha", "alpha", 0.5, 0.0, 1.0, 0.1)]),
    Effect("op_mul", "Multiplicação (A * B)", "operation", "operacao",
           "Operação entre duas imagens: produto normalizado A*B/255 (escurece, "
           "efeito 'multiply'). Sem parâmetros.",
           op_multiply),
    Effect("op_diff", "Diferença absoluta |A - B|", "operation", "operacao",
           "Operação entre duas imagens: |A - B| pixel a pixel (destaca o que mudou). "
           "Sem parâmetros.",
           op_absdiff),
    Effect("op_max", "Clarear (max(A, B))", "operation", "operacao",
           "Operação entre duas imagens: máximo pixel a pixel (efeito 'lighten'). "
           "Sem parâmetros.",
           op_lighten),
]

EFFECTS_BY_KEY: Dict[str, Effect] = {e.key: e for e in EFFECTS}


# --------------------------------------------------------------------------
# Aplicação com seleção de canal
# --------------------------------------------------------------------------
def extract_channel(img: np.ndarray, channel: str) -> np.ndarray:
    """Extrai 'GRAY', 'R', 'G' ou 'B' como imagem 2D contígua."""
    if img.ndim == 2:
        return img
    if channel == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(img[:, :, _BGR_INDEX[channel]])


def fit_to(img: np.ndarray, shape) -> np.ndarray:
    """Redimensiona `img` para (H, W) de `shape` e garante 3 canais BGR."""
    h, w = shape[:2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def apply_effect(img: np.ndarray, effect: Effect, params: Optional[Dict] = None,
                 channel: str = "RGB", other: Optional[np.ndarray] = None) -> np.ndarray:
    """Aplica `effect` em `img` (BGR uint8) e devolve sempre BGR uint8.

    channel = "RGB"  : usa a imagem colorida inteira;
              "GRAY" : converte para cinza antes de filtrar;
              "R/G/B": filtra somente aquele canal e exibe o resultado do canal
                       (imagem em tons de cinza representando o canal filtrado).
    `other` é a segunda imagem, obrigatória para operações.
    """
    params = effect.default_params() if params is None else params
    if effect.kind == "operation":
        if other is None:
            raise ValueError("A operação '%s' precisa de uma segunda imagem." % effect.name)
        other = fit_to(other, img.shape)

    if channel == "RGB":
        out = effect.func(img, other, **params) if effect.kind == "operation" \
            else effect.func(img, **params)
        out = _to_u8(out)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR) if out.ndim == 2 else out

    src = extract_channel(img, channel)
    if effect.kind == "operation":
        out = effect.func(src, extract_channel(other, channel), **params)
    elif effect.color_only:
        out = _gray(effect.func(cv2.cvtColor(src, cv2.COLOR_GRAY2BGR), **params))
    else:
        out = effect.func(src, **params)
    return cv2.cvtColor(_to_u8(out), cv2.COLOR_GRAY2BGR)
