#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сборщик методического PDF про маркет-импакт и аудит расчёта β.

Запуск (в env lobs5, НЕ на login-ноде — см. run_methodology_pdf.sh):
    python make_methodology_pdf.py --out results/market_impact_methodology.pdf

Архитектура:
  * Контент — в content.py (список «блоков»: h1/h2/p/eq/bullets/table/spacer/pagebreak).
  * Формулы (английский LaTeX-синтаксис) рендерятся в PNG через matplotlib MATHTEXT (без полного LaTeX).
  * Кириллица — через шрифт DejaVuSans (лежит внутри пакета matplotlib).
  * Вёрстка: основной путь — fpdf2 (красивый text-flow); fallback — matplotlib PdfPages (нулевые доп. deps).

Скрипт самодостаточен по зависимостям: matplotlib + numpy обязательны; fpdf2 — желателен (иначе fallback).
"""
import argparse
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.image as mpimg         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import content as C                       # noqa: E402

# --- шрифты DejaVu (Cyrillic) из пакета matplotlib --------------------------------
FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
FONT_REG = os.path.join(FONT_DIR, "DejaVuSans.ttf")
FONT_BLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")
FONT_ITL = os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")

# matplotlib по умолчанию использует DejaVu Sans (есть кириллица) — закрепим для подписей/формул
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"


# ---------------------------------------------------------------------------------
# Рендер одной формулы в PNG. На любой сбой mathtext — fallback на «голый» текст,
# чтобы сборка PDF никогда не падала целиком из-за одной формулы.
# ---------------------------------------------------------------------------------
def render_equation(tex, path, fontsize=15, dpi=200):
    def _save(s, mathmode):
        fig = plt.figure(figsize=(0.1, 0.1))
        txt = ("$" + s + "$") if mathmode else s
        fig.text(0.0, 0.0, txt, fontsize=fontsize)
        fig.savefig(path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)

    try:
        _save(tex, True)
    except Exception as e:                       # mathtext не осилил макрос — рисуем как обычный текст
        sys.stderr.write("[eq] mathtext failed (%s); plain fallback for: %s\n" % (e, tex[:60]))
        plain = (tex.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
                    .replace("\\quad", "   ").replace("\\qquad", "      ")
                    .replace("\\left", "").replace("\\right", "")
                    .replace("\\mathrm", "").replace("\\frac", "")
                    .replace("{", "").replace("}", "").replace("\\", ""))
        try:
            plt.close("all")
            _save(plain, False)
        except Exception:
            _save(" ", False)
    arr = mpimg.imread(path)
    h_px, w_px = arr.shape[0], arr.shape[1]
    return w_px / float(h_px)                      # aspect (w/h)


# =================================================================================
# ПУТЬ 1 — fpdf2
# =================================================================================
def build_with_fpdf(blocks, out_path, tmpdir):
    from fpdf import FPDF

    PAGE_W, MARGIN = 210.0, 18.0
    CONTENT_W = PAGE_W - 2 * MARGIN

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_font("DejaVu", "", FONT_REG)
    pdf.add_font("DejaVu", "B", FONT_BLD)
    pdf.add_font("DejaVu", "I", FONT_ITL)
    pdf.set_margins(MARGIN, 16, MARGIN)

    def text_block(s, style="", size=11, h=5.4, gap=1.8):
        pdf.set_font("DejaVu", style, size)
        pdf.multi_cell(CONTENT_W, h, s)
        pdf.ln(gap)

    def wrap(s, width_mm, style, size):
        """Разбить строку на строки по ширине столбца (для таблиц).
        Длинные «неразрывные» токены (напр. beta_sigma1_by_model) тоже режутся по символам."""
        pdf.set_font("DejaVu", style, size)
        avail = width_mm - 2.0
        # сначала бьём слишком длинные токены посимвольно
        toks = []
        for w in s.split(" "):
            if pdf.get_string_width(w) <= avail or len(w) <= 1:
                toks.append(w)
                continue
            piece = ""
            for ch in w:
                if pdf.get_string_width(piece + ch) <= avail or not piece:
                    piece += ch
                else:
                    toks.append(piece)
                    piece = ch
            if piece:
                toks.append(piece)
        lines, cur = [], ""
        for w in toks:
            trial = (cur + " " + w).strip()
            if pdf.get_string_width(trial) <= avail or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines or [""]

    def add_table(headers, rows, widths, size=8.5):
        # масштабируем заданные ширины под CONTENT_W
        scale = CONTENT_W / float(sum(widths))
        widths = [w * scale for w in widths]
        line_h = size * 0.46

        def draw_row(cells, style, fill):
            wrapped = [wrap(str(c), widths[i], style, size) for i, c in enumerate(cells)]
            n = max(len(w) for w in wrapped)
            row_h = n * line_h + 2.0
            if pdf.get_y() + row_h > pdf.page_break_trigger:
                pdf.add_page()
            x0, y0 = pdf.get_x(), pdf.get_y()
            x = x0
            for i, lines in enumerate(wrapped):
                pdf.set_xy(x, y0)
                if fill:
                    pdf.set_fill_color(228, 233, 240)
                    pdf.rect(x, y0, widths[i], row_h, style="F")
                pdf.rect(x, y0, widths[i], row_h)
                pdf.set_font("DejaVu", style, size)
                pad = (row_h - n * line_h) / 2.0
                for j, ln in enumerate(lines):
                    pdf.set_xy(x + 1.0, y0 + pad + j * line_h)
                    pdf.cell(widths[i] - 2.0, line_h, ln)
                x += widths[i]
            pdf.set_xy(x0, y0 + row_h)

        draw_row(headers, "B", True)
        for r in rows:
            draw_row(r, "", False)
        pdf.ln(3)

    def add_equation(tex, idx):
        png = os.path.join(tmpdir, "eq_%03d.png" % idx)
        aspect = render_equation(tex, png)
        target_h = 7.0
        w = target_h * aspect
        if w > CONTENT_W - 4:
            w = CONTENT_W - 4
            target_h = w / aspect
        if pdf.get_y() + target_h + 6 > pdf.page_break_trigger:
            pdf.add_page()
        x = MARGIN + (CONTENT_W - w) / 2.0
        pdf.ln(2)
        pdf.image(png, x=x, y=pdf.get_y(), w=w, h=target_h)
        pdf.set_y(pdf.get_y() + target_h + 4)

    # --- титул ---
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("DejaVu", "B", 19)
    pdf.multi_cell(CONTENT_W, 9, C.TITLE, align="C")
    pdf.ln(3)
    pdf.set_font("DejaVu", "", 13)
    pdf.multi_cell(CONTENT_W, 7, C.SUBTITLE, align="C")
    pdf.ln(8)
    pdf.set_font("DejaVu", "I", 11)
    pdf.multi_cell(CONTENT_W, 6, "Методический обзор и аудит • " + C.DATE, align="C")

    eq_idx = 0
    for blk in blocks:
        t = blk["type"]
        if t == "pagebreak":
            pdf.add_page()
        elif t == "spacer":
            pdf.ln(4)
        elif t == "h1":
            pdf.ln(3)
            text_block(blk["text"], style="B", size=15, h=7.0, gap=2.2)
        elif t == "h2":
            text_block(blk["text"], style="B", size=12.5, h=6.2, gap=1.6)
        elif t == "p":
            text_block(blk["text"], style="", size=11, h=5.4, gap=2.4)
        elif t == "bullets":
            pdf.set_font("DejaVu", "", 10.5)
            for it in blk["items"]:
                pdf.multi_cell(CONTENT_W, 5.2, "•  " + it)
                pdf.ln(1.0)
            pdf.ln(1.6)
        elif t == "eq":
            add_equation(blk["tex"], eq_idx)
            eq_idx += 1
        elif t == "table":
            add_table(blk["headers"], blk["rows"], blk["widths"])
    pdf.output(out_path)


# =================================================================================
# ПУТЬ 2 — matplotlib PdfPages (fallback, нулевые доп. зависимости)
# =================================================================================
def build_with_matplotlib(blocks, out_path, tmpdir):
    from matplotlib.backends.backend_pdf import PdfPages
    import textwrap

    PW, PH = 8.27, 11.69          # A4 в дюймах
    LM, RM, TM, BM = 0.9, 0.9, 0.9, 0.9
    cw = PW - LM - RM

    pdf = PdfPages(out_path)
    state = {"fig": None, "y": 0.0}

    def new_page():
        if state["fig"] is not None:
            pdf.savefig(state["fig"])
            plt.close(state["fig"])
        fig = plt.figure(figsize=(PW, PH))
        state["fig"], state["y"] = fig, PH - TM

    def ensure(space):
        if state["fig"] is None or state["y"] - space < BM:
            new_page()

    def fx(x_in):
        return x_in / PW

    def fy(y_in):
        return y_in / PH

    def put_text(s, size=10, weight="normal", style="normal", wrapn=None, indent=0.0, dy=0.16):
        wrapn = wrapn or max(10, int((cw - indent) / (size * 0.0095)))
        for para in s.split("\n"):
            for ln in (textwrap.wrap(para, wrapn) or [""]):
                ensure(dy + 0.02)
                state["fig"].text(fx(LM + indent), fy(state["y"]), ln,
                                  fontsize=size, fontweight=weight, fontstyle=style,
                                  ha="left", va="top")
                state["y"] -= dy
        state["y"] -= 0.04

    def put_equation(tex, idx):
        png = os.path.join(tmpdir, "eqf_%03d.png" % idx)
        aspect = render_equation(tex, png, fontsize=15, dpi=220)
        h_in = 0.28
        w_in = h_in * aspect
        if w_in > cw - 0.4:
            w_in = cw - 0.4
            h_in = w_in / aspect
        ensure(h_in + 0.18)
        arr = mpimg.imread(png)
        ax = state["fig"].add_axes([fx(LM + (cw - w_in) / 2.0),
                                    fy(state["y"] - h_in), w_in / PW, h_in / PH])
        ax.imshow(arr)
        ax.axis("off")
        state["y"] -= (h_in + 0.16)

    def put_table(headers, rows, size=8):
        put_text(" | ".join(headers), size=size, weight="bold", wrapn=int(cw / (size * 0.0085)))
        for r in rows:
            put_text("  " + " | ".join(str(c) for c in r), size=size,
                     wrapn=int(cw / (size * 0.0085)))
        state["y"] -= 0.04

    # титул
    new_page()
    state["y"] = PH / 1.7
    put_text(C.TITLE, size=17, weight="bold", wrapn=46)
    put_text(C.SUBTITLE, size=12, wrapn=64)
    put_text("Методический обзор и аудит • " + C.DATE, size=10, style="italic")
    new_page()

    eq_idx = 0
    for blk in blocks:
        t = blk["type"]
        if t == "pagebreak":
            new_page()
        elif t == "spacer":
            state["y"] -= 0.15
        elif t == "h1":
            state["y"] -= 0.12
            put_text(blk["text"], size=14, weight="bold")
        elif t == "h2":
            put_text(blk["text"], size=12, weight="bold")
        elif t == "p":
            put_text(blk["text"], size=10)
        elif t == "bullets":
            for it in blk["items"]:
                put_text("•  " + it, size=9.5, indent=0.15)
        elif t == "eq":
            put_equation(blk["tex"], eq_idx)
            eq_idx += 1
        elif t == "table":
            put_table(blk["headers"], blk["rows"])
    if state["fig"] is not None:
        pdf.savefig(state["fig"])
        plt.close(state["fig"])
    pdf.close()


# ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "market_impact_methodology.pdf"))
    ap.add_argument("--engine", choices=["auto", "fpdf", "matplotlib"], default="auto")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for f in (FONT_REG, FONT_BLD, FONT_ITL):
        if not os.path.exists(f):
            sys.stderr.write("ERROR: шрифт не найден: %s\n" % f)
            sys.exit(2)

    blocks = C.get_blocks()
    engine = args.engine
    if engine == "auto":
        try:
            import fpdf  # noqa: F401
            engine = "fpdf"
        except Exception:
            engine = "matplotlib"

    tmpdir = tempfile.mkdtemp(prefix="methpdf_")
    print("[build] engine=%s  out=%s" % (engine, args.out))
    if engine == "fpdf":
        try:
            build_with_fpdf(blocks, args.out, tmpdir)
        except Exception as e:
            sys.stderr.write("[build] fpdf failed (%s) -> matplotlib fallback\n" % e)
            build_with_matplotlib(blocks, args.out, tmpdir)
    else:
        build_with_matplotlib(blocks, args.out, tmpdir)

    size_kb = os.path.getsize(args.out) / 1024.0
    print("[build] OK: %s (%.1f KB)" % (args.out, size_kb))


if __name__ == "__main__":
    main()
