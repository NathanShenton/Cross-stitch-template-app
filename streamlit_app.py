# app.py
import io
from typing import Tuple, List
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Page ----------
st.set_page_config(page_title="Image → Cross-Stitch Chart", layout="wide")
st.title("🧵 Image → Cross-Stitch Chart (Aida)")
st.caption("Upload an image and generate a cross-stitch grid. Export PNG/PDF/CSV.")

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Pattern Settings")
    aida_count = st.number_input("Aida count (stitches per inch)", min_value=6, max_value=28, value=14, step=1)

    units = st.radio("Size input", ["Target stitches", "Target inches (stitched area)"], index=1)
    if units == "Target stitches":
        target_w_st = st.number_input("Width (stitches)", min_value=20, max_value=1000, value=150, step=10)
        target_h_st = st.number_input("Height (stitches) (0 keeps aspect)", min_value=0, max_value=1000, value=0, step=10)
        target_inches = None
    else:
        width_in = st.number_input("Width (inches)", min_value=1.0, max_value=60.0, value=7.0, step=0.5)
        height_in = st.number_input("Height (inches) (0 keeps aspect)", min_value=0.0, max_value=60.0, value=0.0, step=0.5)
        target_inches = (width_in, height_in)

    chart_style = st.selectbox("Chart style", ["Colour blocks", "Mono symbol chart"], index=1)

    # Colour-mode options
    max_colors = st.slider("Max colours (colour mode)", min_value=2, max_value=60, value=25, step=1)
    use_dither = st.checkbox("Floyd–Steinberg dither (colour mode)", value=False)

    # Mono options
    bw_threshold = st.slider("Mono threshold (darker → stitched)", 0.0, 1.0, 0.55, 0.01)

    # Grid look
    show_10_grid = st.checkbox("Bold every 10 stitches", value=True)
    number_every = st.selectbox("Numbering interval", [5, 10], index=1)
    cell_px = st.slider("Cell render size (pixels)", min_value=6, max_value=30, value=14, step=1)
    line_width = st.select_slider("Grid line weight", options=[0.3, 0.5, 0.7, 1.0, 1.5], value=0.5)
    ten_line_width = st.select_slider("10-grid weight", options=[1.0, 1.5, 2.0, 2.5], value=2.0)
    dot_in_mono = st.checkbox("White dot in stitched cells (mono)", value=True)

uploaded = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("Upload an image to get started.")
    st.stop()

img = Image.open(uploaded).convert("RGB")

# ---------- Helpers ----------
def compute_target_stitches(im: Image.Image) -> Tuple[int, int]:
    w, h = im.size
    aspect = w / h
    if target_inches is None:
        tw = int(round(target_w_st))
        th = int(round(tw / aspect)) if target_h_st == 0 else int(round(target_h_st))
    else:
        win, hin = target_inches
        if hin == 0:
            tw = int(round(win * aida_count))
            th = int(round(tw / aspect))
        else:
            tw = int(round(win * aida_count))
            th = int(round(hin * aida_count))
    tw = max(10, min(2000, tw))
    th = max(10, min(2000, th))
    return tw, th

def rgb_to_hex(rgb: List[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def rgb_to_luma(rgb_arr: np.ndarray) -> np.ndarray:
    r, g, b = rgb_arr[..., 0] / 255.0, rgb_arr[..., 1] / 255.0, rgb_arr[..., 2] / 255.0
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def to_mono_stitches(img_rgb: Image.Image, w: int, h: int, threshold: float) -> np.ndarray:
    small = img_rgb.resize((w, h), Image.Resampling.LANCZOS)
    arr = np.array(small, dtype=np.uint8)
    luma = rgb_to_luma(arr)
    return (luma < threshold).astype(np.uint8)  # 1=stitched (black), 0=empty (white)

# ---------- Renderers ----------
def render_color_chart(idx: np.ndarray, palette: np.ndarray, cell_px=12) -> Image.Image:
    H, W = idx.shape
    fig_w = W * cell_px / 100
    fig_h = H * cell_px / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.imshow(palette[idx], interpolation="nearest")
    ax.set_xticks(np.arange(-.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, H, 1), minor=True)
    ax.grid(which="minor", linewidth=line_width, color="0.75")
    if show_10_grid:
        ax.set_xticks(np.arange(-.5, W, 10))
        ax.set_yticks(np.arange(-.5, H, 10))
        ax.grid(which="both", linewidth=ten_line_width, color="0.2")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def render_bw_chart(stitches: np.ndarray, cell_px=14, bold_every=10, number_every=10,
                    line_w=0.5, bold_w=2.0, dot=True) -> Image.Image:
    H, W = stitches.shape
    fig_w = W * cell_px / 100
    fig_h = H * cell_px / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.set_facecolor("white")

    # Filled squares: 1=black, 0=white
    ax.imshow(stitches, cmap="gray_r", interpolation="nearest")

    # Optional small white dot in the centre of each stitched square
    if dot:
        ys, xs = np.where(stitches == 1)
        ax.scatter(xs, ys, s=(cell_px * 0.35), c="white", marker="o")

    # Minor grid each stitch
    ax.set_xticks(np.arange(-.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, H, 1), minor=True)
    ax.grid(which="minor", linewidth=line_w, color="0.75")

    # Bold every N
    if bold_every and bold_every > 0:
        ax.set_xticks(np.arange(-.5, W, bold_every))
        ax.set_yticks(np.arange(-.5, H, bold_every))
        ax.grid(which="both", linewidth=bold_w, color="0.2")

    # Edge numbers
    ax.set_xlim([-0.5, W - 0.5]); ax.set_ylim([H - 0.5, -0.5])
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.tick_params(left=False, bottom=False)
    for x in range(0, W, number_every):
        ax.text(x, -1.2, f"{x}", ha="center", va="top", fontsize=8)
        ax.text(x, H,   f"{x}", ha="center", va="bottom", fontsize=8)
    for y in range(0, H, number_every):
        ax.text(-1.2, y, f"{y}", ha="right", va="center", fontsize=8)
        ax.text(W,   y, f"{y}", ha="left",  va="center", fontsize=8)

    # Neat border
    ax.add_patch(plt.Rectangle((-0.5, -0.5), W, H, fill=False, lw=bold_w, ec="0.1"))
    ax.set_axis_off()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ---------- Main processing ----------
target_w, target_h = compute_target_stitches(img)
stitched_w_in = target_w / aida_count
stitched_h_in = target_h / aida_count

# Prepare chart + optional legend depending on mode
legend = None
symbol_grid = None

if chart_style == "Mono symbol chart":
    stitches_bw = to_mono_stitches(img, target_w, target_h, bw_threshold)
    chart_img = render_bw_chart(
        stitches_bw,
        cell_px=cell_px,
        bold_every=10 if show_10_grid else 0,
        number_every=number_every,
        line_w=line_width,
        bold_w=ten_line_width,
        dot=dot_in_mono
    )
    # Simple two-colour "legend"
    legend = pd.DataFrame(
        {"Symbol": ["■", " "], "Meaning": ["Stitch (black square)", "Empty (white)"]}
    )
    symbol_grid = pd.DataFrame(stitches_bw.astype(int))
else:
    # Colour path: resize → quantise → palette index map
    resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    dither_mode = Image.Dither.FLOYDSTEINBERG if use_dither else Image.Dither.NONE
    quant = resized.convert("P", palette=Image.ADAPTIVE, colors=max_colors, dither=dither_mode).convert("RGB")

    arr = np.array(quant)
    palette_cols, counts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(-counts)
    palette_cols = palette_cols[order]
    counts = counts[order]

    palette_index = {tuple(rgb): i for i, rgb in enumerate(palette_cols.tolist())}
    H, W = arr.shape[:2]
    pattern_idx = np.zeros((H, W), dtype=np.int32)
    for y in range(H):
        for x in range(W):
            pattern_idx[y, x] = palette_index[tuple(arr[y, x])]

    chart_img = render_color_chart(pattern_idx, palette_cols, cell_px=cell_px)

    legend = pd.DataFrame({
        "Symbol": [chr(65 + i) if i < 26 else f"S{i}" for i in range(len(palette_cols))],
        "RGB": [tuple(map(int, c)) for c in palette_cols],
        "Hex": [rgb_to_hex(c) for c in palette_cols],
        "Stitch Count": [int(counts[i]) for i in range(len(palette_cols))]
    })
    legend["% Coverage"] = (legend["Stitch Count"] / legend["Stitch Count"].sum() * 100).round(1)

    # symbols grid for CSV (for colour charts)
    symbols = legend["Symbol"].tolist()
    symbol_grid = pd.DataFrame(np.vectorize(lambda i: symbols[i])(pattern_idx))

# ---------- Layout ----------
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Original (scaled preview)")
    st.image(img, use_column_width=True)
    st.caption(
        f"Target grid: **{target_w}×{target_h} stitches** "
        f"≈ **{stitched_w_in:.2f} × {stitched_h_in:.2f} inches** on {aida_count}-count Aida."
    )
with col2:
    st.subheader("Cross-stitch chart")
    st.image(chart_img, use_column_width=True)

st.subheader("Legend")
st.dataframe(legend, use_container_width=True)

# ---------- Downloads ----------
# Chart PNG
png_buf = io.BytesIO()
chart_img.save(png_buf, format="PNG")
png_bytes = png_buf.getvalue()

# Simple PDF (embed PNG)
pdf_buf = io.BytesIO()
chart_img.save(pdf_buf, format="PDF")
pdf_bytes = pdf_buf.getvalue()

# Pattern CSV (symbols or 0/1 for mono)
csv_buf = io.StringIO()
symbol_grid.to_csv(csv_buf, index=False, header=False)
csv_bytes = csv_buf.getvalue().encode("utf-8")

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("⬇️ Download Chart (PNG)", data=png_bytes, file_name="crossstitch_chart.png", mime="image/png")
with c2:
    st.download_button("⬇️ Download Chart (PDF)", data=pdf_bytes, file_name="crossstitch_chart.pdf", mime="application/pdf")
with c3:
    fn = "pattern_symbols.csv" if chart_style == "Colour blocks" else "pattern_binary_0_1.csv"
    st.download_button("⬇️ Download Pattern CSV", data=csv_bytes, file_name=fn, mime="text/csv")

st.markdown(
    """
**Notes**
- Each pixel = one full cross. Choose stitches or inches; inches convert using your Aida count.
- *Mono symbol chart* matches classic printed patterns (black squares, optional centre dots, bold 10×10, edge numbers).
- *Colour blocks* uses adaptive palette quantisation; adjust **Max colours** and **Dither** to taste.
"""
)
