"""Mechanism figure: recursive consolidation across a task sequence.

Each task trains a three-part decomposition of the CURRENT effective weight
(frozen residual + slow principal + fresh fast adapter); at the boundary the
trained weight is consolidated by SVD and re-split for the next task.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE = "#2a78d6"      # slow principal component
ORANGE = "#eb6834"    # fast adapter
GRAY_F = "#e4e4e4"    # residual fill
GRAY_E = "#9a9a9a"    # residual edge
GREEN = "#008300"     # consolidation arrows
INK = "#222222"

fig, ax = plt.subplots(figsize=(9.8, 2.9))
ax.set_xlim(-1.15, 11.9)
ax.set_ylim(-0.75, 3.35)
ax.axis("off")

W, GAP = 2.0, 1.7
LAYERS = [  # (y0, height, fill, edge, lw, ls, label, text color)
    (0.00, 0.75, GRAY_F, GRAY_E, 1.0, "-", "residual $W_{\\mathrm{res}}^{(t)}$ (frozen)", "#555555"),
    (0.85, 0.75, "#d7e6f8", BLUE, 1.4, "-", "slow $W_{\\mathrm{slow}}^{(t)}$ (tiny lr)", BLUE),
    (1.70, 0.60, "#fdeadf", ORANGE, 1.4, "--", "fast $\\Delta W_{\\mathrm{fast}}^{(t)}\\!\\approx 0$", ORANGE),
]

for t in range(3):
    x0 = t * (W + GAP)
    for y0, h, fc, ec, lw, ls, label, tc in LAYERS:
        ax.add_patch(FancyBboxPatch((x0, y0), W, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls))
        ax.text(x0 + W / 2, y0 + h / 2, label, ha="center", va="center", fontsize=8.2, color=tc)
    ax.text(x0 + W / 2, -0.38, f"train on task $D_{t + 1}$", ha="center", va="center",
            fontsize=8.5, color=INK)
    ax.text(x0 + W / 2, 2.62, f"$W_{{\\mathrm{{eff}}}}^{{({t})}}$" + ("$\\,=W_0$" if t == 0 else ""),
            ha="center", va="center", fontsize=9, color=INK)

    if t < 2:
        xa = x0 + W + 0.06
        ax.add_patch(FancyArrowPatch((xa, 1.25), (xa + GAP - 0.12, 1.25),
                                     arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.6, color=GREEN))
        ax.text(xa + (GAP - 0.06) / 2, 1.95, "consolidate,\nSVD re-split\n(elbow rank $k_{t+1}$)",
                ha="center", va="center", fontsize=7.3, color=GREEN)

ax.text(-1.05, 1.25,
        "$W_{\\mathrm{eff}}^{(t)} =$\n$W_{\\mathrm{res}}^{(t)}$\n$+\\,W_{\\mathrm{slow}}^{(t)}$\n$+\\,\\Delta W_{\\mathrm{fast}}^{(t)}$",
        ha="left", va="center", fontsize=8.2, color=INK)

ax.text(11.15, 2.0, "$\\cdots$", ha="center", va="center", fontsize=16, color=INK)
ax.text(11.15, 1.55, "next task starts from the\nconsolidated model,\nnever from $W_0$ again",
        ha="center", va="top", fontsize=7.6, color=INK, style="italic")

fig.tight_layout()
out = __file__.replace(".py", ".pdf")
fig.savefig(out, bbox_inches="tight")
print("Saved", out)
