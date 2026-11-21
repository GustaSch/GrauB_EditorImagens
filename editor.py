"""Lógica do editor (independente da janela): estado, pipeline, teclas e HUD."""
from __future__ import annotations

import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from filters import CHANNELS, EFFECTS, Effect, apply_effect
from imageio_utils import limit_size, read_image, write_image
from stickers import (PlacedSticker, Sticker, draw_sticker, load_stickers, overlay_rgba,
                      scale_to_width)

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUTPUT = ROOT / "output"

PHOTO_MAX = (1000, 700)   # tamanho máximo de trabalho no modo foto
VIDEO_MAX = (640, 480)    # tamanho máximo de trabalho no modo vídeo
WINDOW = "Editor de Imagens e Video - Grau B"
MIN_CANVAS_W = 780
HUD_H = 78

HELP = """
=========================== COMANDOS ===========================
 l  carregar foto (digite o caminho no console)
 i  carregar a 2ª imagem usada nas operações aritméticas
 v  iniciar/parar a webcam
 n / b   próximo / anterior efeito      t  listar todos os efeitos
 p       próximo parâmetro do efeito    + / -  aumenta / diminui parâmetro
 m       canal: RGB > GRAY > R > G > B  (R/G/B: mostra o resultado do canal)
 a (Enter/Espaço)  aplicar o efeito selecionado
 c       alternar encadeamento (ligado: um filtro sobre o resultado do anterior)
 z       desfazer última ação (filtro ou sticker)
 1-9     escolher sticker (0 = nenhum) e CLIQUE na foto para posicioná-lo
 , / .   diminuir / aumentar o tamanho do sticker
 f       filtro de gatinho por detecção de rosto (foto e vídeo)
 s       salvar imagem/quadro atual em output/
 r       resetar (limpa filtros, stickers e volta à imagem original)
 h       ajuda            ESC ou q   sair
================================================================
"""


def ascii_text(s: str) -> str:
    """cv2.putText só desenha ASCII: remove acentos."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


@dataclass
class Step:
    effect: Effect
    params: Dict[str, float]
    channel: str


class Editor:
    def __init__(self, camera_index: int = 0, sticker_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        self.camera_index = camera_index
        self.output_dir = Path(output_dir) if output_dir else OUTPUT

        self.effects = EFFECTS
        self.effect_idx = 0
        self.param_idx = 0
        self.param_values: Dict[str, Dict[str, float]] = {e.key: e.default_params() for e in EFFECTS}
        self.channel_idx = 0
        self.chain = False

        self.original: Optional[np.ndarray] = None
        self.image2: Optional[np.ndarray] = None
        self.cap = None

        self.steps: List[Step] = []
        self.placed: List[PlacedSticker] = []
        self.history: List[str] = []  # "step" | "sticker"

        self.stickers: List[Sticker] = self._load_stickers(sticker_dir or ASSETS / "stickers")
        self.sticker_sel: Optional[int] = None
        self.sticker_scale = 0.18

        self.face_on = False
        self._cascade = None
        self._face_assets: Optional[Dict[str, np.ndarray]] = None
        self._faces_photo: List[tuple] = []
        self._frame_count = 0
        self._faces_video: List[tuple] = []

        self._cache: Optional[np.ndarray] = None  # resultado dos filtros (modo foto)
        self._dirty = True
        self._img2_cache: Optional[np.ndarray] = None
        self._t0 = time.time()
        self.last_output: Optional[np.ndarray] = None
        self.message = ""

    # ------------------------------------------------------------------ util
    @staticmethod
    def _load_stickers(folder: Path) -> List[Sticker]:
        stickers = load_stickers(folder)
        if not stickers:
            try:
                from generate_assets import generate_all
                generate_all(verbose=False)
                stickers = load_stickers(folder)
            except Exception as exc:  # pragma: no cover
                print("[aviso] não foi possível gerar stickers:", exc)
        return stickers[:9]

    @property
    def effect(self) -> Effect:
        return self.effects[self.effect_idx]

    @property
    def params(self) -> Dict[str, float]:
        return self.param_values[self.effect.key]

    @property
    def channel(self) -> str:
        return CHANNELS[self.channel_idx]

    @property
    def mode(self) -> str:
        if self.cap is not None:
            return "video"
        return "photo" if self.original is not None else "none"

    def say(self, text: str) -> None:
        self.message = text
        print(text)

    # ------------------------------------------------------------- carregar
    def load_photo(self, path) -> bool:
        path = str(path).strip().strip('"').strip("'")
        if not path:
            return False
        img = read_image(path, cv2.IMREAD_COLOR)
        if img is None:
            self.say("Erro: não foi possível abrir a imagem '%s'." % path)
            return False
        self.stop_webcam()
        self.original = limit_size(img, *PHOTO_MAX)
        self._dirty = True
        self._faces_photo = self._detect_faces(self.original) if self.face_on else []
        self.say("Foto carregada: %s (%dx%d)" % (Path(path).name, self.original.shape[1],
                                                 self.original.shape[0]))
        return True

    def load_second_image(self, path) -> bool:
        path = str(path).strip().strip('"').strip("'")
        if not path:
            return False
        img = read_image(path, cv2.IMREAD_COLOR)
        if img is None:
            self.say("Erro: não foi possível abrir a imagem '%s'." % path)
            return False
        self.image2 = img
        self._img2_cache = None
        self._dirty = True
        self.say("2ª imagem (operações aritméticas) carregada: %s" % Path(path).name)
        return True

    def ensure_second_image(self) -> Optional[np.ndarray]:
        if self.image2 is None:
            default = ASSETS / "samples" / "exemplo2.jpg"
            img = read_image(default, cv2.IMREAD_COLOR) if default.exists() else None
            if img is None:  # gradiente de reserva
                h, w = 480, 640
                img = np.zeros((h, w, 3), np.uint8)
                img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
                img[:, :, 2] = np.linspace(255, 0, h, dtype=np.uint8)[:, None]
            self.image2 = img
        return self.image2

    def _second_for(self, shape) -> np.ndarray:
        """2ª imagem já no tamanho do quadro atual (com cache)."""
        img2 = self.ensure_second_image()
        h, w = shape[:2]
        if self._img2_cache is None or self._img2_cache.shape[:2] != (h, w):
            self._img2_cache = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
        return self._img2_cache

    # --------------------------------------------------------------- webcam
    def toggle_webcam(self) -> None:
        if self.cap is not None:
            self.stop_webcam()
            self.say("Webcam parada.")
            return
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened() and sys.platform.startswith("win"):
            cap.release()
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            self.say("Erro: não foi possível abrir a webcam (índice %d). "
                     "Use --camera N para trocar." % self.camera_index)
            return
        self.cap = cap
        self._img2_cache = None
        self._frame_count = 0
        self._faces_video = []
        self.say("Webcam iniciada (imagem espelhada).")

    def stop_webcam(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self._img2_cache = None
            self._dirty = True

    def _read_frame(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.say("Aviso: falha ao ler a webcam; modo vídeo encerrado.")
            self.stop_webcam()
            return None
        return limit_size(cv2.flip(frame, 1), *VIDEO_MAX)

    # ------------------------------------------------------------- pipeline
    def _run_steps(self, img: np.ndarray) -> np.ndarray:
        out = img
        for st in self.steps:
            other = self._second_for(out.shape) if st.effect.kind == "operation" else None
            out = apply_effect(out, st.effect, st.params, st.channel, other)
        return out

    def apply_current(self) -> None:
        if self.mode == "none":
            self.say("Carregue uma foto (l) ou inicie a webcam (v) antes de aplicar.")
            return
        step = Step(self.effect, dict(self.params), self.channel)
        if self.chain:
            self.steps.append(step)
        else:
            self.steps = [step]
            self.history = [h for h in self.history if h != "step"]
        self.history.append("step")
        self._dirty = True
        self.say("Aplicado: %s | canal %s%s" % (self.effect.name, self.channel,
                                                 " | passos: %d" % len(self.steps) if self.chain else ""))

    def undo(self) -> None:
        if not self.history:
            self.say("Nada para desfazer.")
            return
        what = self.history.pop()
        if what == "step" and self.steps:
            self.steps.pop()
            self._dirty = True
        elif what == "sticker" and self.placed:
            self.placed.pop()
        self.say("Última ação desfeita.")

    def reset(self) -> None:
        self.steps.clear()
        self.placed.clear()
        self.history.clear()
        self.face_on = False
        self.sticker_sel = None
        self.channel_idx = 0
        self.param_values = {e.key: e.default_params() for e in EFFECTS}
        self._dirty = True
        self.say("Tudo resetado: voltando à imagem original.")

    # ------------------------------------------------------------- seleção
    def describe(self) -> None:
        e = self.effect
        print("\n[%d/%d] %s  (%s)" % (self.effect_idx + 1, len(self.effects), e.name,
                                      "operação com 2 imagens" if e.kind == "operation" else "filtro"))
        print("  " + e.description)
        if e.params:
            for i, p in enumerate(e.params):
                mark = "*" if i == self.param_idx else " "
                print("  %s %s = %s   (%s; de %s a %s, passo %s)" % (
                    mark, p.name, self.params[p.name], p.label, p.min, p.max, p.step))
        else:
            print("  (sem parâmetros)")

    def select(self, delta: int) -> None:
        self.effect_idx = (self.effect_idx + delta) % len(self.effects)
        self.param_idx = 0
        self.describe()

    def list_effects(self) -> None:
        print("\nEfeitos disponíveis:")
        for i, e in enumerate(self.effects):
            print("  %2d. %-32s [%s]" % (i + 1, e.name, e.category))

    def next_param(self) -> None:
        if not self.effect.params:
            self.say("Este efeito não tem parâmetros.")
            return
        self.param_idx = (self.param_idx + 1) % len(self.effect.params)
        self.describe()

    def adjust(self, direction: int) -> None:
        if not self.effect.params:
            self.say("Este efeito não tem parâmetros.")
            return
        p = self.effect.params[self.param_idx]
        self.params[p.name] = p.clamp(self.params[p.name] + direction * p.step)
        self.say("%s = %s" % (p.name, self.params[p.name]))

    def cycle_channel(self) -> None:
        self.channel_idx = (self.channel_idx + 1) % len(CHANNELS)
        self.say("Canal selecionado: %s" % self.channel)

    def select_sticker(self, idx: Optional[int]) -> None:
        if idx is None:
            self.sticker_sel = None
            self.say("Nenhum sticker selecionado.")
        elif 0 <= idx < len(self.stickers):
            self.sticker_sel = idx
            self.say("Sticker '%s' selecionado: clique na foto para posicionar." %
                     self.stickers[idx].name)
        else:
            self.say("Não existe sticker %d (há %d)." % (idx + 1, len(self.stickers)))

    # -------------------------------------------------------------- mouse
    def on_mouse(self, event, x, y, flags=0, param=None) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.mode != "photo":
            if self.mode == "video":
                self.say("Stickers por clique funcionam no modo foto (pare a webcam com 'v').")
            return
        if self.sticker_sel is None:
            self.say("Escolha um sticker (teclas 1-%d) antes de clicar." % max(1, len(self.stickers)))
            return
        h, w = self.original.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        self.placed.append(PlacedSticker(self.stickers[self.sticker_sel], int(x), int(y),
                                         self.sticker_scale))
        self.history.append("sticker")
        self.say("Sticker '%s' posicionado em (%d, %d)." % (self.stickers[self.sticker_sel].name, x, y))

    # ----------------------------------------------------- detecção de face
    def _get_cascade(self):
        if self._cascade is None:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(path)
            self._cascade = cascade if not cascade.empty() else False
        return self._cascade

    def _load_face_assets(self) -> Optional[Dict[str, np.ndarray]]:
        if self._face_assets is None:
            assets = {}
            for key in ("orelhas", "focinho"):
                img = read_image(ASSETS / "stickers" / "face" / (key + ".png"), cv2.IMREAD_UNCHANGED)
                if img is not None and img.ndim == 3 and img.shape[2] == 4:
                    assets[key] = img
            self._face_assets = assets if len(assets) == 2 else {}
        return self._face_assets or None

    def _detect_faces(self, img: np.ndarray) -> List[tuple]:
        cascade = self._get_cascade()
        if not cascade:
            return []
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                         minSize=(max(30, img.shape[1] // 14),) * 2)
        return [tuple(int(v) for v in f) for f in faces]

    def toggle_face(self) -> None:
        if not self._get_cascade() or not self._load_face_assets():
            self.say("Filtro de rosto indisponível (classificador Haar ou assets não encontrados).")
            return
        self.face_on = not self.face_on
        if self.face_on and self.original is not None:
            self._faces_photo = self._detect_faces(self.original)
            if not self._faces_photo:
                self.say("Filtro de gatinho ligado, mas nenhum rosto foi detectado na foto.")
                return
        self.say("Filtro de gatinho: %s" % ("LIGADO" if self.face_on else "desligado"))

    @staticmethod
    def draw_cat_face(img: np.ndarray, face: tuple, assets: Dict[str, np.ndarray]) -> None:
        x, y, w, h = face
        ears = scale_to_width(assets["orelhas"], w * 1.15)
        overlay_rgba(img, ears, x + w // 2, int(y + 0.12 * h - ears.shape[0] / 2.0))
        nose = scale_to_width(assets["focinho"], w * 0.95)
        overlay_rgba(img, nose, x + w // 2, int(y + 0.60 * h))

    # ------------------------------------------------------------ composição
    def compose(self) -> np.ndarray:
        """Imagem final (sem HUD). Placeholder se nada estiver carregado."""
        t = time.time() - self._t0
        if self.mode == "video":
            frame = self._read_frame()
            if frame is None:
                return self.compose()
            out = self._run_steps(frame).copy()
            if self.face_on and self._load_face_assets():
                if self._frame_count % 3 == 0:
                    self._faces_video = self._detect_faces(frame)
                for f in self._faces_video:
                    self.draw_cat_face(out, f, self._face_assets)
            self._frame_count += 1
            self.last_output = out
            return out
        if self.mode == "photo":
            if self._dirty or self._cache is None:
                self._cache = self._run_steps(self.original)
                self._dirty = False
            out = self._cache.copy()
            if self.face_on and self._face_assets:
                for f in self._faces_photo:
                    self.draw_cat_face(out, f, self._face_assets)
            for p in self.placed:
                draw_sticker(out, p, t)
            self.last_output = out
            return out
        blank = np.full((420, MIN_CANVAS_W, 3), 35, np.uint8)
        for i, line in enumerate(("Nenhuma imagem carregada.",
                                  "Pressione 'l' para carregar uma foto ou 'v' para a webcam.",
                                  "Pressione 'h' para ver todos os comandos.")):
            cv2.putText(blank, line, (30, 180 + 36 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (230, 230, 230), 1, cv2.LINE_AA)
        self.last_output = None
        return blank

    def render(self) -> np.ndarray:
        """Imagem composta + painel HUD (o painel fica abaixo, sem cobrir a imagem)."""
        img = self.compose()
        h, w = img.shape[:2]
        cw = max(w, MIN_CANVAS_W)
        canvas = np.full((h + HUD_H, cw, 3), 28, np.uint8)
        canvas[:h, :w] = img
        e = self.effect
        ps = "  ".join(("*" if i == self.param_idx else "") + "%s=%s" % (p.name, self.params[p.name])
                        for i, p in enumerate(e.params)) or "(sem parametros)"
        sel = self.stickers[self.sticker_sel].name if self.sticker_sel is not None else "-"
        lines = [
            "[%s] %s | canal: %s | encadear: %s | passos: %d" % (
                self.mode.upper(), e.name, self.channel, "SIM" if self.chain else "nao", len(self.steps)),
            "params: %s" % ps,
            "stickers: %d | sel: %s (%.0f%%) | rosto: %s | h = ajuda" % (
                len(self.placed), sel, self.sticker_scale * 100, "ON" if self.face_on else "off"),
        ]
        for i, line in enumerate(lines):
            cv2.putText(canvas, ascii_text(line), (10, h + 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (235, 235, 235), 1, cv2.LINE_AA)
        return canvas

    # ---------------------------------------------------------------- salvar
    def save(self) -> Optional[Path]:
        if self.last_output is None and self.mode == "photo":
            self.compose()
        if self.last_output is None:
            self.say("Nada para salvar: carregue uma foto ou inicie a webcam.")
            return None
        prefix = "frame" if self.mode == "video" else "foto"
        path = self.output_dir / ("%s_%s.png" % (prefix, datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
        if write_image(path, self.last_output):
            self.say("Imagem salva em: %s" % path)
            return path
        self.say("Erro ao salvar em %s" % path)
        return None

    # ---------------------------------------------------------------- teclas
    def handle_key(self, key: int) -> bool:
        """Processa uma tecla. Retorna False para encerrar o programa."""
        if key in (-1, 255):
            return True
        if key in (27, ord("q")):
            return False
        if key in (13, 10, 32, ord("a")):
            self.apply_current()
        elif key == ord("n"):
            self.select(+1)
        elif key == ord("b"):
            self.select(-1)
        elif key == ord("t"):
            self.list_effects()
        elif key == ord("p"):
            self.next_param()
        elif key in (ord("+"), ord("=")):
            self.adjust(+1)
        elif key in (ord("-"), ord("_")):
            self.adjust(-1)
        elif key == ord("m"):
            self.cycle_channel()
        elif key == ord("c"):
            self.chain = not self.chain
            self.say("Encadeamento de filtros: %s" % ("LIGADO" if self.chain else "desligado"))
        elif key == ord("z"):
            self.undo()
        elif key == ord("r"):
            self.reset()
        elif key == ord("s"):
            self.save()
        elif key == ord("v"):
            self.toggle_webcam()
        elif key == ord("l"):
            self.load_photo(self._ask("Caminho da foto (Enter cancela): "))
        elif key == ord("i"):
            self.load_second_image(self._ask("Caminho da 2ª imagem (Enter cancela): "))
        elif key == ord("f"):
            self.toggle_face()
        elif key == ord("h"):
            print(HELP)
        elif key == ord("0"):
            self.select_sticker(None)
        elif ord("1") <= key <= ord("9"):
            self.select_sticker(key - ord("1"))
        elif key == ord(","):
            self.sticker_scale = max(0.04, round(self.sticker_scale - 0.02, 2))
            self.say("Tamanho do sticker: %.0f%% da largura" % (self.sticker_scale * 100))
        elif key == ord("."):
            self.sticker_scale = min(0.8, round(self.sticker_scale + 0.02, 2))
            self.say("Tamanho do sticker: %.0f%% da largura" % (self.sticker_scale * 100))
        return True

    @staticmethod
    def _ask(prompt: str) -> str:
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            return ""
