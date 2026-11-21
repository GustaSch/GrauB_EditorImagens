"""Editor de imagens e vídeo (estilo stories) — Processamento Gráfico: Fundamentos, Grau B.

Uso:
    python main.py                       # abre o editor vazio (use 'l' ou 'v')
    python main.py --image foto.jpg      # já carrega uma foto
    python main.py --camera 1            # escolhe a webcam (padrão 0)
"""
from __future__ import annotations

import argparse
import sys

import cv2

from editor import HELP, WINDOW, Editor


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Editor de imagens e vídeo com filtros e stickers (OpenCV).")
    ap.add_argument("--image", help="foto inicial a carregar")
    ap.add_argument("--image2", help="2ª imagem para as operações aritméticas")
    ap.add_argument("--camera", type=int, default=0, help="índice da webcam (padrão: 0)")
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    editor = Editor(camera_index=args.camera)
    print(HELP)
    if args.image2:
        editor.load_second_image(args.image2)
    if args.image:
        editor.load_photo(args.image)
    editor.describe()

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, editor.on_mouse)

    frames = 0
    try:
        while True:
            cv2.imshow(WINDOW, editor.render())
            frames += 1
            # No modo vídeo o próprio read() da webcam limita o ritmo.
            key = cv2.waitKey(1 if editor.mode == "video" else 30) & 0xFF
            if not editor.handle_key(key):
                break
            if frames > 5 and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # janela fechada pelo "X"
    except KeyboardInterrupt:
        pass
    finally:
        editor.stop_webcam()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args()))
