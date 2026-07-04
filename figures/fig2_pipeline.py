"""
Figure 2: ReCoLoRA pipeline — an intuitive, story-style diagram.

Reads left to right as four steps:
  (1) a frozen pretrained weight matrix W0
  (2) split it by importance: SVD spectrum + elbow cut (principal vs residual)
  (3) two-stage training: learn principal first, then ease in residual via gamma(t)
  (4) the resulting continual model

Real charts (spectrum bars, gamma ramp) replace abstract boxes so the idea is
visible at a glance instead of described only in text.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch

# ── palette ───────────────────────────────────────────────────────────────────
BLUE,  LBLUE  = "#2B6CB0", "#DBEAFE"
ORANGE, LORNG = "#C05621", "#FFEDD5"
GREEN, LGRN   = "#276749", "#DCFCE7"
GRAY,  LGRAY  = "#A0AEC0", "#E2E8F0"
DGRAY         = "#2D3748"
BG            = "#FFFFFF"

FIGW, FIGH = 14.0, 4.8
fig = plt.figure(figsize=(FIGW, FIGH), facecolor=BG)
bg = fig.add_axes([0, 0, 1, 1])
bg.set_xlim(0, FIGW)
bg.set_ylim(0, FIGH)
bg.axis('off')

def fx(x): return x / FIGW
def fy(y): return y / FIGH

Y_MID = 2.55          # vertical centre of the flow
Y_HEAD = 4.50         # numbered step circles
Y_TITLE = 4.08        # step titles
Y_CAP = 1.18          # plain-language captions

def header(cx, num, title, color):
    bg.add_patch(Circle((cx, Y_HEAD), 0.17, facecolor=color,
                        edgecolor='none', zorder=6))
    bg.text(cx, Y_HEAD, str(num), ha='center', va='center',
            color='white', fontsize=12, fontweight='bold', zorder=7)
    bg.text(cx, Y_TITLE, title, ha='center', va='center',
            color=color, fontsize=12.5, fontweight='bold')

def caption(cx, text, color=DGRAY):
    bg.text(cx, Y_CAP, text, ha='center', va='center',
            fontsize=9.2, color=color, linespacing=1.3)

def arrow(x1, x2, y=Y_MID, color=DGRAY):
    bg.add_patch(FancyArrowPatch((x1, y), (x2, y),
                 arrowstyle='-|>', mutation_scale=22,
                 lw=2.4, color=color, zorder=8,
                 shrinkA=0, shrinkB=0))

# ── panel centres ───────────────────────────────────────────────────────────────
CX1, CX2, CX3, CX4 = 1.55, 4.75, 8.65, 12.25

# ════════════════════════════════════════════════════════════════════════════════
# Step 1 — frozen pretrained weight matrix
# ════════════════════════════════════════════════════════════════════════════════
header(CX1, 1, "Pretrained weight", GRAY)

np.random.seed(7)
n, cell = 5, 0.21
gx0 = CX1 - n * cell / 2
gy0 = Y_MID - n * cell / 2 + 0.05
vals = np.random.rand(n, n)
for i in range(n):
    for j in range(n):
        bg.add_patch(Rectangle((gx0 + j * cell, gy0 + i * cell),
                               cell * 0.9, cell * 0.9,
                               facecolor=plt.cm.Greys(0.20 + 0.55 * vals[i, j]),
                               edgecolor='white', lw=0.6, zorder=4))
bg.text(CX1, gy0 - 0.30, r"$W_0$  (frozen)", ha='center', va='center',
        fontsize=11.5, fontweight='bold', color=DGRAY)
caption(CX1, "the original\nbackbone weight")

arrow(CX1 + 0.95, CX2 - 1.85)

# ════════════════════════════════════════════════════════════════════════════════
# Step 2 — split by importance: SVD spectrum + elbow cut
# ════════════════════════════════════════════════════════════════════════════════
header(CX2, 2, "Split by importance", BLUE)

axL, axR, axB, axT = 3.30, 6.20, 1.62, 3.55
ax2 = fig.add_axes([fx(axL), fy(axB), fx(axR - axL), fy(axT - axB)])
sig = np.array([1.00, 0.80, 0.62, 0.47, 0.33,
                0.19, 0.155, 0.13, 0.112, 0.10,
                0.09, 0.082, 0.076, 0.071])
r_star = 5
colors = [BLUE if k < r_star else GRAY for k in range(len(sig))]
ax2.bar(range(len(sig)), sig, color=colors, width=0.78, zorder=3)
ax2.axvline(r_star - 0.5, color=ORANGE, ls=(0, (4, 3)), lw=2.0, zorder=4)
ax2.annotate(r"elbow $r^{*}$",
             xy=(r_star - 0.5, 0.86), xytext=(r_star + 1.4, 0.92),
             fontsize=10, color=ORANGE, fontweight='bold',
             ha='center',
             arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.6))
from matplotlib.patches import Patch
ax2.legend(handles=[Patch(facecolor=BLUE, label="principal (keep)"),
                    Patch(facecolor=GRAY, label="residual")],
           loc='upper right', bbox_to_anchor=(1.0, 0.66),
           fontsize=9.2, frameon=False,
           handlelength=1.1, handleheight=1.1, borderaxespad=0.2,
           labelspacing=0.5)
ax2.set_xlim(-0.7, len(sig) - 0.3)
ax2.set_ylim(0, 1.05)
ax2.set_xlabel("singular value index", fontsize=8.8, labelpad=2)
ax2.set_ylabel(r"$\sigma$", fontsize=10, rotation=0, labelpad=8)
ax2.set_xticks([])
ax2.set_yticks([])
for s in ('top', 'right'):
    ax2.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax2.spines[s].set_color(GRAY)
caption(CX2, "randomized SVD ranks directions;\nthe elbow keeps the few dominant ones")

arrow(axR + 0.10, CX3 - 1.95)

# ════════════════════════════════════════════════════════════════════════════════
# Step 3 — two-stage training: principal first, then residual via gamma(t)
# ════════════════════════════════════════════════════════════════════════════════
header(CX3, 3, "Two-stage training", ORANGE)

bxL, bxR, bxB, bxT = 7.15, 10.20, 1.62, 3.55
ax3 = fig.add_axes([fx(bxL), fy(bxB), fx(bxR - bxL), fy(bxT - bxB)])
t = np.linspace(0, 1, 300)
tau = 0.5
gamma = np.clip((t - tau) / 0.32, 0, 1)
ax3.axvspan(0, tau, color=LBLUE, alpha=0.7, zorder=1)
ax3.axvspan(tau, 1, color=LORNG, alpha=0.7, zorder=1)
ax3.plot(t, gamma, color=ORANGE, lw=2.6, zorder=4)
ax3.axvline(tau, color=DGRAY, ls=(0, (3, 3)), lw=1.4, zorder=3)
ax3.text(tau, 1.12, r"boundary $\tau$", ha='center', fontsize=8.6,
         color=DGRAY)
ax3.text(0.25, 0.55, "Stage 1\nprincipal\n$\\Delta W_p$", ha='center',
         va='center', fontsize=9.0, color=BLUE, fontweight='bold',
         linespacing=1.25)
ax3.text(0.78, 0.30, "Stage 2\n+ residual\n$\\gamma(t)\\Delta W_r$",
         ha='center', va='center', fontsize=9.0, color=ORANGE,
         fontweight='bold', linespacing=1.25)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1.05)
ax3.set_xlabel("training progress", fontsize=8.8, labelpad=2)
ax3.set_ylabel(r"$\gamma(t)$", fontsize=10, rotation=0, labelpad=8)
ax3.set_xticks([])
ax3.set_yticks([0, 1])
ax3.tick_params(labelsize=8, colors=GRAY)
for s in ('top', 'right'):
    ax3.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax3.spines[s].set_color(GRAY)
caption(CX3, "learn dominant structure first,\nthen ease in extra capacity")

arrow(bxR + 0.10, CX4 - 1.55)

# ════════════════════════════════════════════════════════════════════════════════
# Step 4 — continual model
# ════════════════════════════════════════════════════════════════════════════════
header(CX4, 4, "Continual model", GREEN)

ow, oh = 3.05, 1.55
bg.add_patch(FancyBboxPatch((CX4 - ow / 2, Y_MID - oh / 2), ow, oh,
                            boxstyle="round,pad=0.10",
                            facecolor=LGRN, edgecolor=GREEN,
                            lw=2.4, zorder=3))
bg.text(CX4, Y_MID + 0.30, "ready to use", ha='center', va='center',
        fontsize=10.5, fontweight='bold', color=GREEN, zorder=4)
bg.text(CX4, Y_MID - 0.28,
        r"$W_{\mathrm{eff}}=W_0+\Delta W_p+\gamma(t)\,\Delta W_r$",
        ha='center', va='center', fontsize=10.5, color=DGRAY, zorder=4)
caption(CX4, "backbone + adapted\nprincipal & residual parts")

# ── save ────────────────────────────────────────────────────────────────────────
for ext in ('png', 'pdf'):
    kw = dict(bbox_inches='tight', facecolor=BG)
    if ext == 'png':
        kw['dpi'] = 200
    plt.savefig(f"/home/bhqy/Documents/project/HiLoRA/figures/fig2_pipeline.{ext}", **kw)
print("Done.")
