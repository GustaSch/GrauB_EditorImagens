"""Testes sem janela/webcam.  Rode:  python -m unittest discover -s tests -v"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import editor as editor_mod  # noqa: E402
import main as main_mod  # noqa: E402
import ui as ui_mod  # noqa: E402
from editor import Editor  # noqa: E402
from filters import CHANNELS, EFFECTS, apply_effect  # noqa: E402
from imageio_utils import read_image, write_image  # noqa: E402
from stickers import (PlacedSticker, draw_sticker, load_stickers, overlay_rgba,  # noqa: E402
                      scale_to_width)

SAMPLE = ROOT / "assets" / "samples" / "exemplo1.jpg"


def rand_img(h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


class FilterTests(unittest.TestCase):
    def test_requirements_counts(self):
        filters = [e for e in EFFECTS if e.kind == "filter"]
        ops = [e for e in EFFECTS if e.kind == "operation"]
        self.assertGreaterEqual(len(filters), 10)
        self.assertGreaterEqual(len(ops), 3)
        cats = {e.category for e in filters}
        self.assertIn("suavizacao", cats)
        self.assertIn("realce", cats)
        for e in EFFECTS:
            self.assertTrue(e.description.strip(), e.name)

    def test_every_effect_every_channel(self):
        img, img2 = rand_img(seed=1), rand_img(seed=2)
        for e in EFFECTS:
            for ch in CHANNELS:
                out = apply_effect(img, e, None, ch, img2 if e.kind == "operation" else None)
                self.assertEqual(out.shape, img.shape, (e.name, ch))
                self.assertEqual(out.dtype, np.uint8, (e.name, ch))
                if ch in ("GRAY", "R", "G", "B"):
                    self.assertTrue(np.array_equal(out[:, :, 0], out[:, :, 1]), (e.name, ch))

    def test_params_at_limits(self):
        img = rand_img()
        for e in EFFECTS:
            if e.kind != "filter":
                continue
            for pick in ("min", "max"):
                params = {p.name: p.clamp(getattr(p, pick)) for p in e.params}
                out = apply_effect(img, e, params, "RGB")
                self.assertEqual(out.shape, img.shape, (e.name, pick, params))

    def test_single_channel_shows_filtered_channel(self):
        img = rand_img(seed=3)
        e = next(x for x in EFFECTS if x.key == "box")
        out = apply_effect(img, e, {"ksize": 5}, "R")
        expected = cv2.blur(img[:, :, 2], (5, 5))
        self.assertTrue(np.array_equal(out[:, :, 0], expected))

    def test_ops_values(self):
        a = np.full((4, 4, 3), 200, np.uint8)
        b = np.full((4, 4, 3), 100, np.uint8)
        by = {e.key: e for e in EFFECTS}
        self.assertTrue((apply_effect(a, by["op_add"], {"weight": 1.0}, "RGB", b) == 255).all())
        self.assertTrue((apply_effect(a, by["op_sub"], {"weight": 1.0}, "RGB", b) == 100).all())
        self.assertTrue((apply_effect(b, by["op_sub"], {"weight": 1.0}, "RGB", a) == 0).all())
        self.assertTrue((apply_effect(a, by["op_blend"], {"alpha": 0.5}, "RGB", b) == 150).all())
        self.assertTrue((apply_effect(a, by["op_diff"], None, "RGB", b) == 100).all())
        self.assertTrue((apply_effect(a, by["op_max"], None, "RGB", b) == 200).all())

    def test_op_resizes_second_image(self):
        e = next(x for x in EFFECTS if x.key == "op_blend")
        out = apply_effect(rand_img(40, 50), e, None, "RGB", rand_img(10, 10))
        self.assertEqual(out.shape, (40, 50, 3))

    def test_op_without_second_image_fails_clearly(self):
        e = next(x for x in EFFECTS if x.key == "op_add")
        with self.assertRaises(ValueError):
            apply_effect(rand_img(), e, None, "RGB", None)

    def test_brightness_negative_not_abs(self):
        e = next(x for x in EFFECTS if x.key == "bc")
        out = apply_effect(np.full((3, 3, 3), 20, np.uint8), e, {"contrast": 1.0, "brightness": -50}, "RGB")
        self.assertTrue((out == 0).all())

    def test_sepia_known_value(self):
        e = next(x for x in EFFECTS if x.key == "sepia")
        px = np.array([[[100, 150, 200]]], np.uint8)  # B,G,R
        out = apply_effect(px, e, None, "RGB")[0, 0]
        r = min(255, int(round(0.393 * 200 + 0.769 * 150 + 0.189 * 100)))
        self.assertAlmostEqual(int(out[2]), r, delta=1)


class StickerTests(unittest.TestCase):
    def test_load_and_alpha(self):
        sts = load_stickers(ROOT / "assets" / "stickers")
        self.assertGreaterEqual(len(sts), 5)
        self.assertTrue(any(s.animated for s in sts))
        for s in sts:
            for f in s.frames:
                self.assertEqual(f.shape[2], 4)
                self.assertTrue((f[:, :, 3] == 0).any(), s.name)    # transparência real
                self.assertTrue((f[:, :, 3] == 255).any(), s.name)  # e opacidade real

    def test_overlay_blend_and_clip(self):
        base = np.zeros((20, 20, 3), np.uint8)
        sprite = np.zeros((10, 10, 4), np.uint8)
        sprite[:, :, 1] = 255
        sprite[:, :, 3] = 255
        overlay_rgba(base, sprite, 10, 10)
        self.assertTrue((base[5:15, 5:15, 1] == 255).all())
        self.assertEqual(int(base[0, 0, 1]), 0)
        # parcialmente fora / totalmente fora não devem dar erro
        overlay_rgba(base, sprite, 0, 0)
        overlay_rgba(base, sprite, 19, 19)
        overlay_rgba(base, sprite, -100, -100)
        overlay_rgba(base, sprite, 500, 500)
        # alfa 50%
        base2 = np.zeros((4, 4, 3), np.uint8)
        sp2 = np.full((4, 4, 4), 255, np.uint8)
        sp2[:, :, 3] = 128
        overlay_rgba(base2, sp2, 2, 2)
        self.assertAlmostEqual(int(base2[1, 1, 0]), 128, delta=1)

    def test_animation_frames(self):
        s = next(x for x in load_stickers(ROOT / "assets" / "stickers") if x.animated)
        self.assertEqual(len(s.frames), 6)
        self.assertFalse(np.array_equal(s.frame_at(0.0), s.frame_at(0.14)))
        self.assertTrue(np.array_equal(s.frame_at(0.0), s.frame_at(6 / s.fps)))

    def test_draw_sticker_tiny_scale(self):
        s = load_stickers(ROOT / "assets" / "stickers")[0]
        base = np.zeros((30, 30, 3), np.uint8)
        draw_sticker(base, PlacedSticker(s, 15, 15, 0.001), 0.0)
        self.assertEqual(scale_to_width(s.frames[0], 0).shape[1], 1)


class EditorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ed = Editor(output_dir=Path(self.tmp.name))
        self.ed.load_photo(SAMPLE)

    def tearDown(self):
        self.tmp.cleanup()

    def key(self, ch):
        return self.ed.handle_key(ord(ch) if isinstance(ch, str) else ch)

    def test_photo_apply_reset_undo(self):
        orig = self.ed.original.copy()
        self.assertTrue(np.array_equal(self.ed.compose(), orig))
        self.key("a")  # aplica o 1º efeito (Box Blur)
        self.assertFalse(np.array_equal(self.ed.compose(), orig))
        self.key("r")
        self.assertTrue(np.array_equal(self.ed.compose(), orig))
        self.key("a")
        self.key("z")
        self.assertTrue(np.array_equal(self.ed.compose(), orig))

    def test_chain_vs_replace(self):
        self.key("a")
        self.key("n")
        self.key("a")
        self.assertEqual(len(self.ed.steps), 1)  # sem encadear: substitui
        self.key("c")
        self.key("n")
        self.key("a")
        self.key("n")
        self.key("a")
        self.assertEqual(len(self.ed.steps), 3)
        self.key("z")
        self.assertEqual(len(self.ed.steps), 2)

    def test_param_adjust_clamped_and_odd(self):
        self.key("+")
        self.assertEqual(self.ed.params["ksize"] % 2, 1)
        for _ in range(100):
            self.key("+")
        self.assertEqual(self.ed.params["ksize"], 31)
        for _ in range(100):
            self.key("-")
        self.assertEqual(self.ed.params["ksize"], 3)
        self.key("p")  # efeito com 1 parâmetro: não quebra

    def test_all_effects_via_keys_all_channels(self):
        for _ in range(len(EFFECTS) * 2):
            for _ in CHANNELS:
                self.key("m")
                self.key("a")
                out = self.ed.compose()
                self.assertEqual(out.shape, self.ed.original.shape)
            self.key("n")
            self.key("p")
            self.key("+")
        self.key("r")
        self.assertEqual(self.ed.channel, "RGB")

    def test_sticker_click_and_undo(self):
        orig = self.ed.original.copy()
        self.ed.on_mouse(cv2.EVENT_LBUTTONDOWN, 50, 50)  # sem sticker selecionado
        self.assertEqual(len(self.ed.placed), 0)
        for k in "123456789":
            self.key(k)
        self.key("1")
        self.ed.on_mouse(cv2.EVENT_LBUTTONDOWN, 200, 150)
        self.ed.on_mouse(cv2.EVENT_LBUTTONDOWN, 10**6, 10)  # fora da imagem
        self.ed.on_mouse(cv2.EVENT_MOUSEMOVE, 20, 20)
        self.assertEqual(len(self.ed.placed), 1)
        self.assertFalse(np.array_equal(self.ed.compose(), orig))
        self.key("7")  # animado
        self.ed.on_mouse(cv2.EVENT_LBUTTONDOWN, 400, 300)
        self.key(",")
        self.key(".")
        self.key("z")
        self.assertEqual(len(self.ed.placed), 1)
        self.key("r")
        self.assertEqual(len(self.ed.placed), 0)
        self.assertTrue(np.array_equal(self.ed.compose(), orig))

    def test_save_roundtrip_unicode_path(self):
        sub = Path(self.tmp.name) / "saída çãõ"
        self.ed.output_dir = sub
        self.key("a")
        saved = self.ed.save()
        self.assertIsNotNone(saved)
        back = read_image(saved)
        self.assertTrue(np.array_equal(back, self.ed.last_output))

    def test_render_has_hud(self):
        canvas = self.ed.render()
        self.assertGreater(canvas.shape[0], self.ed.original.shape[0])
        self.assertTrue(np.array_equal(canvas[:self.ed.original.shape[0], :self.ed.original.shape[1]],
                                       self.ed.original))

    def test_empty_editor_is_safe(self):
        ed = Editor(output_dir=Path(self.tmp.name))
        self.assertEqual(ed.mode, "none")
        for k in "anbpmcz+-,.rfstl0123456789h":
            if k in "li":
                continue
            ed.handle_key(ord(k))
        ed.on_mouse(cv2.EVENT_LBUTTONDOWN, 5, 5)
        self.assertEqual(ed.render().ndim, 3)
        self.assertIsNone(ed.save())

    def test_bad_paths(self):
        self.assertFalse(self.ed.load_photo("/nao/existe.jpg"))
        self.assertFalse(self.ed.load_photo(""))
        txt = Path(self.tmp.name) / "x.txt"
        txt.write_text("não sou imagem")
        self.assertFalse(self.ed.load_photo(txt))
        self.assertFalse(self.ed.load_second_image("/nao/existe.jpg"))
        self.assertEqual(self.ed.mode, "photo")

    def test_operation_uses_second_image(self):
        self.key("c")
        while self.ed.effect.key != "op_blend":
            self.key("n")
        self.key("a")
        out = self.ed.compose()
        self.assertEqual(out.shape, self.ed.original.shape)
        self.assertFalse(np.array_equal(out, self.ed.original))
        # 2ª imagem de outro tamanho
        write_image(Path(self.tmp.name) / "b.png", rand_img(17, 23))
        self.assertTrue(self.ed.load_second_image(Path(self.tmp.name) / "b.png"))
        self.assertEqual(self.ed.compose().shape, self.ed.original.shape)

    def test_face_cat_overlay(self):
        base = np.full((300, 300, 3), 128, np.uint8)
        self.ed.toggle_face()  # não há rosto na cena sintética: não pode quebrar
        self.assertTrue(self.ed._load_face_assets())
        before = base.copy()
        Editor.draw_cat_face(base, (100, 120, 100, 100), self.ed._face_assets)
        self.assertFalse(np.array_equal(base, before))
        Editor.draw_cat_face(base, (-20, -20, 100, 100), self.ed._face_assets)  # recorte nas bordas
        self.ed.compose()

    def test_cascade_loads_bundled_xml(self):
        # O XML versionado em assets/ garante o filtro de rosto mesmo no OpenCV 5.x,
        # cujo pacote não traz os classificadores Haar.
        self.assertTrue((editor_mod.ASSETS / editor_mod.CASCADE_FILE).exists())
        self.assertTrue(self.ed._get_cascade())

    def test_cascade_from_accented_path(self):
        folder = Path(self.tmp.name) / "pasta com acentuação"
        folder.mkdir()
        dst = folder / editor_mod.CASCADE_FILE
        dst.write_bytes((editor_mod.ASSETS / editor_mod.CASCADE_FILE).read_bytes())
        self.assertIsNotNone(Editor._read_cascade(dst))
        self.assertIsNone(Editor._read_cascade(folder / "nao_existe.xml"))

    def test_reset_restores_chain_and_sticker_scale(self):
        self.key("c")
        self.key(".")
        self.assertTrue(self.ed.chain)
        self.key("r")
        self.assertFalse(self.ed.chain)
        self.assertEqual(self.ed.sticker_scale, editor_mod.STICKER_SCALE_DEFAULT)

    def test_undo_without_history_says_so(self):
        self.ed.history.clear()
        self.ed.undo()
        self.assertIn("Nada para desfazer", self.ed.message)

    def test_hud_shows_effect_description(self):
        canvas = self.ed.render()
        h = self.ed.original.shape[0]
        area = canvas[h + editor_mod.HUD_BASE_H: h + editor_mod.HUD_H, :]
        self.assertGreater(int((area != 28).any(axis=2).sum()), 50)

    def test_descriptions_fit_in_hud(self):
        import textwrap
        for e in self.ed.effects:
            lines = textwrap.wrap(editor_mod.ascii_text(e.description), editor_mod.HUD_DESC_WIDTH)
            self.assertLessEqual(len(lines), editor_mod.HUD_DESC_LINES, e.name)

    def test_quit_keys(self):
        self.assertTrue(self.key("a"))
        self.assertFalse(self.key(27))
        self.assertFalse(self.key("q"))


class InterfaceTests(unittest.TestCase):
    """Carrossel, botões, sliders e paleta de stickers (cliques simulados, sem janela)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ed = Editor(output_dir=Path(self.tmp.name))
        self.ed.load_photo(SAMPLE)
        self.canvas = self.ed.render()

    def tearDown(self):
        self.tmp.cleanup()

    def hits(self, kind):
        return [h for h in self.ed._hits if h[4] == kind]

    def click(self, hit, event=cv2.EVENT_LBUTTONDOWN, dx=0):
        self.ed.on_mouse(event, (hit[0] + hit[2]) // 2 + dx, (hit[1] + hit[3]) // 2)

    def rerender(self):
        self.canvas = self.ed.render()

    def test_layout_and_toolbar_fit(self):
        self.assertEqual(self.canvas.shape[1], max(self.ed.original.shape[1], editor_mod.MIN_CANVAS_W)
                         + ui_mod.LEGEND_W)
        self.assertEqual(self.canvas.shape[0], self.ed.original.shape[0] + editor_mod.HUD_H + ui_mod.CHROME_H)
        cw = self.canvas.shape[1] - ui_mod.LEGEND_W
        for h in self.ed._hits:
            self.assertLessEqual(h[2], cw, h)  # nada passa da borda (nem a barra de botões)
        self.assertEqual(len(self.hits("key")), 11)

    def test_carousel_click_selects_and_double_click_applies(self):
        target = [h for h in self.hits("effect") if h[5] != self.ed.effect_idx][0]
        self.click(target)
        self.assertEqual(self.ed.effect_idx, target[5])
        self.assertEqual(len(self.ed.steps), 0)
        self.click(target, cv2.EVENT_LBUTTONDBLCLK)
        self.assertEqual(len(self.ed.steps), 1)
        self.assertEqual(self.ed.steps[0].effect, self.ed.effects[target[5]])

    def test_carousel_arrows_and_window_follows_selection(self):
        self.rerender()
        self.click(self.hits("page")[1])  # seta ">"
        self.assertGreater(self.ed.effect_idx, 1)
        for i in range(len(self.ed.effects)):
            self.ed.select_index(i)
            self.rerender()
            self.assertIn(i, [h[5] for h in self.hits("effect")])
        self.click(self.hits("page")[1])
        self.assertEqual(self.ed.effect_idx, len(self.ed.effects) - 1)

    def test_buttons_repeat_shortcuts(self):
        by_key = {h[5]: h for h in self.hits("key")}
        self.click(by_key[ord("a")])
        self.assertEqual(len(self.ed.steps), 1)
        self.click(by_key[ord("z")])
        self.assertEqual(len(self.ed.steps), 0)
        self.click(by_key[ord("m")])
        self.assertEqual(self.ed.channel, "GRAY")
        self.click(by_key[ord("c")])
        self.assertTrue(self.ed.chain)
        self.click(by_key[ord("r")])
        self.assertFalse(self.ed.chain)
        self.assertEqual(self.ed.channel, "RGB")
        self.click(by_key[ord("s")])
        self.assertTrue(list(Path(self.tmp.name).glob("foto_*.png")))

    def test_load_buttons_queue_dialogs(self):
        by_key = {h[5]: h for h in self.hits("key")}
        with mock.patch.object(Editor, "_pick_file", staticmethod(lambda t: str(SAMPLE))):
            self.click(by_key[ord("i")])
            self.assertEqual(self.ed.pending_keys, [ord("i")])  # não abre diálogo dentro do clique
            self.assertTrue(self.ed.process_queue())
        self.assertEqual(self.ed.pending_keys, [])
        self.assertIn("2ª imagem", self.ed.message)

    def test_slider_click_drag_and_release(self):
        while len(self.ed.effect.params) < 3:  # Bilateral: 3 parâmetros
            self.ed.select(+1)
        self.rerender()
        sl = {h[5]: h for h in self.hits("slider")}
        self.assertEqual(sorted(sl), [0, 1, 2])
        x0, x1 = sl[1][6]
        p = self.ed.effect.params[1]
        self.ed.on_mouse(cv2.EVENT_LBUTTONDOWN, x0, sl[1][1] + 5)
        self.assertEqual(self.ed.param_idx, 1)
        self.assertEqual(self.ed.params[p.name], p.min)
        self.ed.on_mouse(cv2.EVENT_MOUSEMOVE, x1, sl[1][1] + 5, cv2.EVENT_FLAG_LBUTTON)
        self.assertEqual(self.ed.params[p.name], p.max)
        self.ed.on_mouse(cv2.EVENT_MOUSEMOVE, (x0 + x1) // 2, 0, cv2.EVENT_FLAG_LBUTTON)
        mid = self.ed.params[p.name]
        self.assertTrue(p.min < mid < p.max)
        self.assertEqual((mid - p.min) % p.step, 0)  # alinhado ao passo
        self.ed.on_mouse(cv2.EVENT_LBUTTONUP, x1, 0)
        self.ed.on_mouse(cv2.EVENT_MOUSEMOVE, x0, 0, cv2.EVENT_FLAG_LBUTTON)  # sem arraste: ignora
        self.assertEqual(self.ed.params[p.name], mid)
        self.ed.on_mouse(cv2.EVENT_LBUTTONDOWN, x1 + 5000, sl[0][1] + 5)  # fora da coluna: clique na foto
        self.assertEqual(self.ed.params[p.name], mid)

    def test_slider_values_respect_param_rules(self):
        for e in EFFECTS:
            for p in e.params:
                for x in (-50, 0, 37, 100, 163, 1000):
                    v = ui_mod.slider_value(p, x, 0, 100)
                    self.assertTrue(p.min <= v <= p.max, (e.name, p.name, x, v))
                    if p.integer and p.step == 2 and p.min % 2 == 1:
                        self.assertEqual(v % 2, 1, (e.name, p.name, x, v))  # ksize sempre ímpar

    def test_param_label_selects_parameter(self):
        while len(self.ed.effect.params) < 2:
            self.ed.select(+1)
        self.rerender()
        self.click([h for h in self.hits("param") if h[5] == 1][0])
        self.assertEqual(self.ed.param_idx, 1)

    def test_sticker_palette(self):
        icons = self.hits("sticker")
        self.assertEqual(len(icons), len(self.ed.stickers))
        self.click(icons[1])
        self.assertEqual(self.ed.sticker_sel, 1)
        self.ed.on_mouse(cv2.EVENT_LBUTTONDOWN, 200, 150)
        self.assertEqual(len(self.ed.placed), 1)
        before = self.ed.sticker_scale
        self.click(self.hits("sticker_size")[1])  # "+"
        self.assertGreater(self.ed.sticker_scale, before)
        self.click(self.hits("sticker_size")[0])  # "-"
        self.click(self.hits("sticker_none")[0])
        self.assertIsNone(self.ed.sticker_sel)

    def test_legend_toggle(self):
        with_legend = self.ed.render().shape[1]
        self.ed.handle_key(ord("h"))
        self.assertFalse(self.ed.show_legend)
        self.assertEqual(self.ed.render().shape[1], with_legend - ui_mod.LEGEND_W)
        self.ed.handle_key(ord("h"))
        self.assertEqual(self.ed.render().shape[1], with_legend)

    def test_thumbnails_every_effect_and_channel(self):
        self.ed.ensure_second_image()
        for ch in range(len(CHANNELS)):
            self.ed.channel_idx = ch
            for e in self.ed.effects:
                th = ui_mod.thumbnail(self.ed, e)
                self.assertEqual(th.shape, (ui_mod.THUMB_H, ui_mod.THUMB_W, 3), (e.name, ch))
        t1 = ui_mod.thumbnail(self.ed, self.ed.effect)
        self.assertIs(ui_mod.thumbnail(self.ed, self.ed.effect), t1)  # cache

    def test_thumbnail_follows_params_and_image(self):
        e = [x for x in self.ed.effects if x.key == "pixelate"][0] if any(
            x.key == "pixelate" for x in self.ed.effects) else self.ed.effects[0]
        a = ui_mod.thumbnail(self.ed, e).copy()
        p = e.params[0]
        self.ed.param_values[e.key][p.name] = p.max
        b = ui_mod.thumbnail(self.ed, e)
        self.assertFalse(np.array_equal(a, b))
        self.ed.load_photo(SAMPLE)
        self.assertEqual(ui_mod.thumbnail(self.ed, e).shape, a.shape)

    def test_render_without_image_and_in_video(self):
        ed = Editor(output_dir=Path(self.tmp.name))
        c = ed.render()  # sem imagem: miniaturas usam a imagem de exemplo
        self.assertEqual(c.shape[1], editor_mod.MIN_CANVAS_W + ui_mod.LEGEND_W)
        with mock.patch.object(cv2, "VideoCapture", FakeCap):
            ed.toggle_webcam()
            c = ed.render()
            self.assertIsNotNone(ed._thumb_snapshot)
            self.assertEqual(ui_mod.thumbnail(ed, ed.effect).shape[:2], (ui_mod.THUMB_H, ui_mod.THUMB_W))
            ed.stop_webcam()
            self.assertIsNone(ed._thumb_snapshot)


class FakeCap:
    """Webcam falsa: gera quadros sintéticos."""
    opened = []
    max_frames = 10**9

    def __init__(self, *a, **k):
        self.n = 0
        self.released = False
        FakeCap.opened.append(self)

    def isOpened(self):
        return True

    def read(self):
        self.n += 1
        if self.n > FakeCap.max_frames:
            return False, None
        rng = np.random.default_rng(self.n)
        return True, rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class VideoAndLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        FakeCap.max_frames = 10**9
        self.tmp.cleanup()

    def test_video_mode(self):
        with mock.patch.object(cv2, "VideoCapture", FakeCap):
            ed = Editor(output_dir=Path(self.tmp.name))
            ed.toggle_webcam()
            self.assertEqual(ed.mode, "video")
            f = ed.compose()
            self.assertLessEqual(f.shape[1], 640)
            for _ in range(len(EFFECTS)):
                for _ in CHANNELS:
                    ed.handle_key(ord("m"))
                    ed.handle_key(ord("a"))
                    self.assertEqual(ed.compose().shape, f.shape)
                ed.handle_key(ord("n"))
            ed.handle_key(ord("r"))
            ed.handle_key(ord("c"))  # encadeamento também no vídeo
            for _ in range(3):
                ed.handle_key(ord("a"))
                ed.handle_key(ord("n"))
            self.assertEqual(len(ed.steps), 3)
            self.assertEqual(ed.compose().shape, f.shape)
            ed.toggle_face()
            ed.compose()
            ed.on_mouse(cv2.EVENT_LBUTTONDOWN, 10, 10)  # ignorado no vídeo
            self.assertEqual(len(ed.placed), 0)
            self.assertIsNotNone(ed.save())
            ed.handle_key(ord("r"))
            FakeCap.max_frames = FakeCap.opened[-1].n + 5
            while ed.mode == "video":  # a webcam falsa acaba: deve voltar sozinho a 'none'
                ed.compose()
            self.assertTrue(FakeCap.opened[-1].released)
            self.assertEqual(ed.mode, "none")

    def test_webcam_fail_message(self):
        class Bad(FakeCap):
            def isOpened(self):
                return False
        with mock.patch.object(cv2, "VideoCapture", Bad):
            ed = Editor(output_dir=Path(self.tmp.name))
            ed.toggle_webcam()
            self.assertEqual(ed.mode, "none")

    def test_main_loop_scripted(self):
        keys = [ord(c) for c in "tnbpm+-ac"] + [ord("v")] + [-1] * 3 + [ord("s"), ord("v")] \
            + [ord("l"), ord("1"), ord("s"), ord("r"), 27]
        shown = []
        it = iter(keys)
        answers = iter([str(SAMPLE)])
        with mock.patch.object(cv2, "namedWindow"), mock.patch.object(cv2, "setMouseCallback"), \
                mock.patch.object(cv2, "imshow", lambda n, img: shown.append(img.shape)), \
                mock.patch.object(cv2, "waitKey", lambda d=0: next(it)), \
                mock.patch.object(cv2, "getWindowProperty", lambda *a: 1.0), \
                mock.patch.object(cv2, "destroyAllWindows"), \
                mock.patch.object(cv2, "VideoCapture", FakeCap), \
                mock.patch.object(Editor, "_pick_file", staticmethod(lambda t: "")), \
                mock.patch.object(Editor, "_ask", staticmethod(lambda p: next(answers))), \
                mock.patch.object(editor_mod, "OUTPUT", Path(self.tmp.name)):
            rc = main_mod.run(main_mod.parse_args(["--image", str(SAMPLE)]))
        self.assertEqual(rc, 0)
        self.assertGreater(len(shown), 10)

    def test_main_window_closed(self):
        with mock.patch.object(cv2, "namedWindow"), mock.patch.object(cv2, "setMouseCallback"), \
                mock.patch.object(cv2, "imshow"), mock.patch.object(cv2, "waitKey", lambda d=0: 255), \
                mock.patch.object(cv2, "getWindowProperty", lambda *a: 0.0), \
                mock.patch.object(cv2, "destroyAllWindows"):
            self.assertEqual(main_mod.run(main_mod.parse_args([])), 0)


if __name__ == "__main__":
    unittest.main()
