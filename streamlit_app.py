import io
import math
from typing import Tuple, List
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image → Cross-Stitch Chart", layout="wide")

st.title("🧵 Image → Cross-Stitch Chart (Aida)")
st.caption("Upload an image and generate a cross-stitch grid + legend. Exports PNG/PDF/CSV.")

# ---- Sidebar controls
with st.sidebar:
    st.header("Pattern Settings")
    aida_count = st.number_input("Aida count (stitches per inch)", min_value=6, max_value=28, value=14, step=1)
    units = st.radio("Size input", ["Target stitches", "Target inches (stitched area)"], index=1)
    if units == "Target stitches":
        target_w_st = st.number_input("Width (stitches)", min_value=20, max_value=800, value=150, step=10)
        target_h_st = st.number_input("Height (stitches)", min_value=20, max_value=800, value=0, step=10,
                                      help="0 keeps aspect ratio from width")
        target_inches = None
    else:
        width_in = st.number_input("Width (inches)", min_value=1.0, max_value=40.0, value=7.0, step=0.5)
        height_in = st.number_input("Height (inches) (0 keeps aspect)", min_value=0.0, max_value=40.0, value=0.0, step=0.5)
        target_inches = (width_in, height_in)

    max_colors = st.slider("Max colours", min_value=2, max_value=60, value=25, step=1)
    use_dither = st.checkbox("Floyd-Steinberg dither", value=False)
    show_10_grid = st.checkbox("Bold every 10 stitches", value=True)
    cell_px = st.slider("Cell render size (pixels)", min_value=6, max_value=30, value=14, step=1)
    line_width = st.select_slider("Grid line weight", options=[0.3, 0.5, 0.7, 1.0, 1.5], value=0.5)
    ten_line_width = st.select_slider("10-grid weight", options=[1.0, 1.5, 2.0, 2.5], value=2.0)

uploaded = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("Upload an image to get started.")
    st.stop()

img = Image.open(uploaded).convert("RGB")

# ---- Compute target grid size
def compute_target_stitches(im: Image.Image) -> Tuple[int, int]:
    w, h = im.size
    aspect = w / h
    if target_inches is None:
        # stitches provided
        tw = int(round(target_w_st))
        th = int(round(tw / aspect)) if target_h_st == 0 else int(round(target_h_st))
    else:
        win, hin = target_inches
        if hin == 0:
            # keep aspect from width
            tw = int(round(win * aida_count))
            th = int(round(tw / aspect))
        else:
            tw = int(round(win * aida_count))
            th = int(round(hin * aida_count))
    # Keep within sane limits
    tw = max(10, min(2000, tw))
    th = max(10, min(2000, th))
    return tw, th

target_w, target_h = compute_target_stitches(img)

# ---- Resize to grid (each pixel = 1 stitch)
resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

# ---- Quantize colours
dither_mode = Image.Dither.FLOYDSTEINBERG if use_dither else Image.Dither.NONE
quant = resized.convert("P", palette=Image.ADAPTIVE, colors=max_colors, dither=dither_mode).convert("RGB")

# Extract palette actually used
arr = np.array(quant)
palette_cols, counts = np.unique(arr.reshape(-1,3), axis=0, return_counts=True)
# Sort by frequency desc
order = np.argsort(-counts)
palette_cols = palette_cols[order]
counts = counts[order]

# Map each pixel to a palette index
def build_palette_index_map(colors: np.ndarray) -> dict:
    # exact RGB match lookup
    return {tuple(rgb): i for i, rgb in enumerate(colors.tolist())}

palette_index = build_palette_index_map(palette_cols)
H, W = arr.shape[:2]
pattern_idx = np.zeros((H, W), dtype=np.int32)
for y in range(H):
    for x in range(W):
        pattern_idx[y, x] = palette_index[tuple(arr[y, x])]

# ---- Build legend dataframe
def rgb_to_hex(rgb: List[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)

legend = pd.DataFrame({
    "Symbol": [chr(65+i) if i < 26 else f"S{i}" for i in range(len(palette_cols))],
    "RGB": [tuple(map(int, c)) for c in palette_cols],
    "Hex": [rgb_to_hex(c) for c in palette_cols],
    "Stitch Count": [int(counts[i]) for i in range(len(palette_cols))]
})
legend["% Coverage"] = (legend["Stitch Count"] / legend["Stitch Count"].sum() * 100).round(1)

# ---- Render chart
def render_chart(idx: np.ndarray, palette: np.ndarray, cell_px=12) -> Image.Image:
    H, W = idx.shape
    fig_w = W * cell_px / 100  # inches at 100 dpi
    fig_h = H * cell_px / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.imshow(palette[idx], interpolation="nearest")
    ax.set_xticks(np.arange(-.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-.5, H, 1), minor=True)
    ax.grid(which="minor", linewidth=line_width)
    if show_10_grid:
        ax.set_xticks(np.arange(-.5, W, 10))
        ax.set_yticks(np.arange(-.5, H, 10))
        ax.grid(which="both", linewidth=ten_line_width)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

chart_img = render_chart(pattern_idx, palette_cols, cell_px=cell_px)

# ---- Layout
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Original (scaled preview)")
    st.image(img, use_column_width=True)
    stitched_w_in = target_w / aida_count
    stitched_h_in = target_h / aida_count
    st.caption(f"Target grid: **{target_w}×{target_h} stitches** "
               f"≈ **{stitched_w_in:.2f} × {stitched_h_in:.2f} inches** on {aida_count}-count Aida.")

with col2:
    st.subheader("Cross-stitch chart")
    st.image(chart_img, use_column_width=True)

st.subheader("Colour legend")
st.dataframe(legend, use_container_width=True)

# ---- Downloads
# Pattern CSV (symbols per cell)
symbols = legend["Symbol"].tolist()
symbol_grid = np.vectorize(lambda i: symbols[i])(pattern_idx)
csv_buf = io.StringIO()
pd.DataFrame(symbol_grid).to_csv(csv_buf, index=False, header=False)
csv_bytes = csv_buf.getvalue().encode("utf-8")

# Chart PNG
png_buf = io.BytesIO()
chart_img.save(png_buf, format="PNG")
png_bytes = png_buf.getvalue()

# Simple PDF export (embed PNG into PDF page)
pdf_buf = io.BytesIO()
chart_img.save(pdf_buf, format="PDF")
pdf_bytes = pdf_buf.getvalue()

dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button("⬇️ Download Chart (PNG)", data=png_bytes, file_name="crossstitch_chart.png", mime="image/png")
with dl2:
    st.download_button("⬇️ Download Chart (PDF)", data=pdf_bytes, file_name="crossstitch_chart.pdf", mime="application/pdf")
with dl3:
    st.download_button("⬇️ Download Pattern CSV (symbols)", data=csv_bytes, file_name="pattern_symbols.csv", mime="text/csv")

st.markdown(
    """
**Notes & tips**
- Each pixel represents **one full cross**. The app resizes your image to your target stitch grid.
- **Max colours** controls how complex the palette is. Try 12–30 for most images.
- **Dithering** can help gradients but may add speckle; many stitchers prefer it off.
- **Every-10 grid** helps counting. Printed PDF uses the on-screen look.
- Want real thread codes (e.g., DMC)? You can add a CSV of thread RGBs and map each palette colour to the nearest thread by ΔE. I can show that next if you want.
"""
)
