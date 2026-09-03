#!/usr/bin/env python3
"""
Generate GPS Trust Index (GTI) analytical graphs for thesis & presentation.

GTI formula:  G_t = α·G_{t-1} + (1-α)·(1 - σ(k·ΔΦ_t - θ))
σ(x) = 1 / (1 + e^{-x})   (logistic sigmoid)

Parameters:
  α  — EMA smoothing factor (0.95 default)
  k  — scaling factor on consistency score (10 default)
  θ  — sigmoid offset / detection threshold (2 default)
  ΔΦ — GPS–IMU consistency score (Euclidean discrepancy in metres)

Outputs written to thesis/figures/ directory.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 200
FS_TITLE = 14
FS_AXIS = 12
FS_TICK = 10
FS_LEGEND = 9

# ─── Deakin brand colours ─────────────────────────────────────────────
DEAKIN_BLUE = "#003399"
DARK_TEAL = "#00667A"
ORANGE = "#E87722"
RED = "#C8102E"
GREEN = "#2D8C4A"
GRAY = "#666666"
LIGHT_BLUE = "#7BAFD4"
LIGHT_ORANGE = "#F6B26B"
BLACK = "#222222"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_AXIS,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_gti(delta_phi_series, alpha=0.95, k=10.0, theta=2.0, g0=1.0):
    """Run GTI over a time series of consistency scores ΔΦ."""
    gti = np.zeros(len(delta_phi_series))
    gti[0] = g0
    for i in range(1, len(gti)):
        susp = sigmoid(k * delta_phi_series[i] - theta)
        trust_reading = 1.0 - susp
        gti[i] = alpha * gti[i - 1] + (1 - alpha) * trust_reading
    return gti


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1 — How the sigmoid suspicion score works
# ═══════════════════════════════════════════════════════════════════════
def fig1_sigmoid_components():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1a — Sigmoid σ(k·x - θ) for different k
    x = np.linspace(0, 5, 500)
    ks = [5, 10, 20, 50]
    colors = [DEAKIN_BLUE, DARK_TEAL, ORANGE, RED]
    ax = axes[0]
    for k, c in zip(ks, colors):
        ax.plot(x, sigmoid(k * x - 2.0), color=c, linewidth=2, label=f"k={k}")
    ax.axvline(2.0 / max(ks), color=GRAY, linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("ΔΦ (GPS–IMU consistency score, m)")
    ax.set_ylabel("σ(k·ΔΦ − θ) — Suspicion Score")
    ax.set_title("Effect of Scaling Factor k\n(θ = 2 fixed)")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.02, 1.05)

    # 1b — Sigmoid σ(k·x - θ) for different θ
    thetas = [0.5, 1.0, 2.0, 3.0]
    ax = axes[1]
    for theta, c in zip(thetas, colors):
        ax.plot(x, sigmoid(10 * x - theta), color=c, linewidth=2, label=f"θ={theta}")
    ax.set_xlabel("ΔΦ (GPS–IMU consistency score, m)")
    ax.set_ylabel("σ(k·ΔΦ − θ) — Suspicion Score")
    ax.set_title("Effect of Offset θ\n(k = 10 fixed)")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.02, 1.05)

    # 1c — Suspicion score -> Trust reading (1 − σ)
    ax = axes[2]
    x2 = np.linspace(0, 5, 500)
    susp = sigmoid(10 * x2 - 2.0)
    ax.fill_between(x2, 0, susp, alpha=0.12, color=RED, label="Suspicion σ(·)")
    ax.plot(x2, susp, color=RED, linewidth=2)
    ax.plot(x2, 1.0 - susp, color=GREEN, linewidth=2, label="Trust reading 1−σ(·)")
    ax.fill_between(x2, 0, 1.0 - susp, alpha=0.08, color=GREEN)
    ax.axvline(2.0 / 10.0, color=GRAY, linestyle=":", linewidth=1, alpha=0.6,
               label=f"θ/k = 0.2")
    ax.set_xlabel("ΔΦ (m)")
    ax.set_title("From Suspicion -> Trust Reading\n(k=10, θ=2)")
    ax.legend(loc="center right")
    ax.set_ylim(-0.02, 1.05)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gti_sigmoid_components.png")
    fig.savefig(path)
    print(f"  ✓ {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Effect of α (smoothing) on GTI decay under sustained spoofing
# ═══════════════════════════════════════════════════════════════════════
def fig2_alpha_sensitivity():
    time_s = np.arange(0, 30.1, 0.1)
    # Spoofing starts at t=3s, constant ΔΦ = 1.5m
    delta_phi = np.where(time_s >= 3.0, 1.5, 0.0)

    alphas = [0.80, 0.90, 0.95, 0.98, 0.99]
    colors = [ORANGE, RED, DEAKIN_BLUE, DARK_TEAL, GREEN]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 2a — GTI decay curves
    for alpha, c in zip(alphas, colors):
        gti = compute_gti(delta_phi, alpha=alpha, k=10.0, theta=2.0)
        ax1.plot(time_s, gti, color=c, linewidth=2, label=f"α = {alpha}")

    ax1.axvspan(3.0, 30.0, alpha=0.06, color=RED)
    ax1.text(9, 0.92, "Spoofing Active", fontsize=9, color=RED, alpha=0.8,
             ha="center", style="italic")
    ax1.axhline(0.7, color=GRAY, linestyle=":", linewidth=1, alpha=0.5)
    ax1.axhline(0.3, color=GRAY, linestyle=":", linewidth=1, alpha=0.5)
    ax1.text(29, 0.72, "G=0.7", fontsize=8, color=GRAY, ha="right")
    ax1.text(29, 0.32, "G=0.3", fontsize=8, color=GRAY, ha="right")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("GPS Trust Index (GTI)")
    ax1.set_title("GTI Decay Under Sustained Spoofing\nΔΦ = 1.5m, k=10, θ=2")
    ax1.legend(loc="lower left")
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_xlim(0, 30)

    # 2b — Time to reach G=0.3 vs α
    alphas_fine = np.linspace(0.75, 0.995, 50)
    t_to_03 = []
    for a in alphas_fine:
        gti = compute_gti(delta_phi, alpha=a, k=10.0, theta=2.0)
        idx = np.argmax(gti <= 0.3)
        t_to_03.append(time_s[idx] if idx > 0 else 30)
    ax2.plot(alphas_fine, t_to_03, color=DEAKIN_BLUE, linewidth=2.5)
    ax2.fill_between(alphas_fine, t_to_03, alpha=0.15, color=DEAKIN_BLUE)
    ax2.scatter([0.95], [t_to_03[np.argmin(np.abs(alphas_fine - 0.95))]],
                color=RED, s=80, zorder=5)
    ax2.annotate(f"α=0.95 -> {t_to_03[np.argmin(np.abs(alphas_fine - 0.95))]:.0f}s",
                 xy=(0.95, t_to_03[np.argmin(np.abs(alphas_fine - 0.95))]),
                 xytext=(0.88, 18), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=BLACK))
    ax2.set_xlabel("Smoothing Factor α")
    ax2.set_ylabel("Time to G ≤ 0.3 (seconds)")
    ax2.set_title("Response Delay vs Smoothing Factor")
    ax2.set_xlim(0.75, 1.0)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gti_alpha_sensitivity.png")
    fig.savefig(path)
    print(f"  ✓ {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3 — Normal flight vs Spoofing — GTI trajectories
# ═══════════════════════════════════════════════════════════════════════
def fig3_normal_vs_spoof():
    time_s = np.arange(0, 30.1, 0.1)

    # Normal flight: ΔΦ fluctuates 0.05–0.4m (wind, sensor noise)
    np.random.seed(42)
    delta_phi_normal = 0.15 + 0.12 * np.sin(2 * np.pi * 0.3 * time_s) + \
                       0.05 * np.random.randn(len(time_s))
    delta_phi_normal = np.abs(delta_phi_normal)

    # Spoofing: t=0-3s normal, then ΔΦ ramps 0.2->2.0m
    delta_phi_spoof = np.where(time_s < 3.0,
                               delta_phi_normal,
                               0.2 + (2.0 - 0.2) * np.minimum(1.0, (time_s - 3.0) / 5.0))

    gti_normal = compute_gti(delta_phi_normal, alpha=0.95, k=10.0, theta=2.0)
    gti_spoof = compute_gti(delta_phi_spoof, alpha=0.95, k=10.0, theta=2.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Top — ΔΦ comparison
    ax1.fill_between(time_s, 0, delta_phi_normal, alpha=0.3, color=GREEN,
                     label="Normal flight (wind + noise)")
    ax1.plot(time_s, delta_phi_spoof, color=RED, linewidth=2,
             label="Spoofing attack")
    ax1.axhline(2.0 / 10.0, color=GRAY, linestyle="--", linewidth=1, alpha=0.6,
                label="θ/k threshold = 0.2m")
    ax1.axvspan(3.0, 30.0, alpha=0.05, color=RED)
    ax1.set_ylabel("ΔΦ — GPS–IMU Consistency Score (m)")
    ax1.set_title("GPS–IMU Inconsistency Over Time")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_xlim(0, 30)

    # Bottom — GTI
    ax2.plot(time_s, gti_normal, color=GREEN, linewidth=2.5,
             label="GTI — Normal flight")
    ax2.plot(time_s, gti_spoof, color=RED, linewidth=2.5,
             label="GTI — Under spoofing")
    # Graduated thresholds
    ax2.axhline(0.7, color=ORANGE, linestyle=":", linewidth=1.2, alpha=0.7)
    ax2.text(29, 0.72, "G=0.7 • Attenuate GPS", fontsize=8, color=ORANGE, ha="right")
    ax2.axhline(0.3, color=RED, linestyle=":", linewidth=1.2, alpha=0.7)
    ax2.text(29, 0.32, "G=0.3 • Re-weight sensors", fontsize=8, color=RED, ha="right")
    ax2.axhline(0.1, color=RED, linestyle="--", linewidth=1, alpha=0.5)
    ax2.text(29, 0.12, "G=0.1 • HOLD mode", fontsize=8, color=RED, ha="right")
    ax2.axhline(0.05, color=RED, linestyle="--", linewidth=1, alpha=0.5)
    ax2.text(29, 0.07, "G=0.05 • Emergency RTL", fontsize=8, color=RED, ha="right")
    ax2.axvspan(3.0, 30.0, alpha=0.05, color=RED)
    ax2.text(9, 0.9, "Attack Active", fontsize=9, color=RED, alpha=0.8,
             ha="center", style="italic")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("GPS Trust Index (GTI)")
    ax2.set_title("GTI: Normal Flight vs GPS Spoofing Attack\n(α=0.95, k=10, θ=2)")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlim(0, 30)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gti_normal_vs_spoof.png")
    fig.savefig(path)
    print(f"  ✓ {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4 — Graduated Response Protocol overlaid on GTI decay
# ═══════════════════════════════════════════════════════════════════════
def fig4_graduated_response():
    time_s = np.arange(0, 25.1, 0.05)
    delta_phi = np.where(time_s < 2.0, 0.1, 1.8)
    gti = compute_gti(delta_phi, alpha=0.95, k=10.0, theta=2.0)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(time_s, gti, color=DEAKIN_BLUE, linewidth=3, zorder=5)

    zones = [
        (0.0, 0.7, GREEN, "GTI ≥ 0.7 — Full GPS Trust", 0),
        (0.7, 0.3, ORANGE, "0.3 ≤ GTI < 0.7 — Attenuate GPS Weight", 1),
        (0.3, 0.1, "#E6A817", "0.1 ≤ GTI < 0.3 — Re-weight Sensors", 2),
        (0.1, 0.05, RED, "0.05 ≤ GTI < 0.1 — HOLD Mode", 3),
        (0.05, 0.0, "#800020", "GTI < 0.05 — Emergency RTL", 4),
    ]

    for lo, hi, color, label, _ in zones:
        ax.fill_between(time_s, lo, hi, alpha=0.08, color=color, linewidth=0)
        ax.axhspan(lo, hi, alpha=0.08, color=color)
        ax.text(25.3, (lo + hi) / 2, label, fontsize=9, color=color,
                va="center", ha="left", fontweight="bold")

    ax.axvspan(2.0, 25.0, alpha=0.05, color=RED)
    ax.text(13, 0.92, "GPS Spoofing Active", fontsize=10, color=RED,
            alpha=0.7, ha="center", style="italic")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("GPS Trust Index (GTI)")
    ax.set_title("Graduated Response Protocol — Escalation as GTI Declines\n(α=0.95, k=10, θ=2, ΔΦ=1.8m)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, 25)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gti_graduated_response.png")
    fig.savefig(path)
    print(f"  ✓ {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5 — Parameter grid: (α, k) heatmap — time to G≤0.3
# ═══════════════════════════════════════════════════════════════════════
def fig5_parameter_heatmaps():
    time_s = np.linspace(0, 60, 600)
    delta_phi = np.where(time_s >= 3.0, 1.5, 0.0)

    alphas = np.linspace(0.80, 0.99, 40)
    ks = np.linspace(2, 30, 40)
    thetas = np.linspace(0.5, 5.0, 40)
    T_AK = np.zeros((len(ks), len(alphas)))
    T_AT = np.zeros((len(thetas), len(alphas)))

    for i, a in enumerate(alphas):
        for j, k in enumerate(ks):
            gti = compute_gti(delta_phi, alpha=a, k=k, theta=2.0)
            idx = np.argmax(gti <= 0.3)
            T_AK[j, i] = time_s[idx] if idx > 0 else 60
        for j, th in enumerate(thetas):
            gti = compute_gti(delta_phi, alpha=a, k=10.0, theta=th)
            idx = np.argmax(gti <= 0.3)
            T_AT[j, i] = time_s[idx] if idx > 0 else 60

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    im1 = ax1.pcolormesh(alphas, ks, T_AK, shading="auto", cmap="RdYlGn_r")
    ax1.scatter([0.95], [10], color=DEAKIN_BLUE, s=120, marker="s", zorder=10,
                edgecolors="white", linewidth=1.5)
    ax1.text(0.952, 10.5, "Default", fontsize=8, color=DEAKIN_BLUE, fontweight="bold")
    ax1.set_xlabel("Smoothing Factor α")
    ax1.set_ylabel("Scaling Factor k")
    ax1.set_title("Time to G ≤ 0.3 (seconds)\nΔΦ = 1.5m, θ = 2")
    cbar1 = fig.colorbar(im1, ax=ax1, label="Seconds", shrink=0.82)
    cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    im2 = ax2.pcolormesh(alphas, thetas, T_AT, shading="auto", cmap="RdYlGn_r")
    ax2.scatter([0.95], [2.0], color=DEAKIN_BLUE, s=120, marker="s", zorder=10,
                edgecolors="white", linewidth=1.5)
    ax2.text(0.952, 2.1, "Default", fontsize=8, color=DEAKIN_BLUE, fontweight="bold")
    ax2.set_xlabel("Smoothing Factor α")
    ax2.set_ylabel("Offset θ")
    ax2.set_title("Time to G ≤ 0.3 (seconds)\nΔΦ = 1.5m, k = 10")
    cbar2 = fig.colorbar(im2, ax=ax2, label="Seconds", shrink=0.82)
    cbar2.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gti_parameter_heatmaps.png")
    fig.savefig(path)
    print(f"  ✓ {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6 — Attack intensity sensitivity (varying ΔΦ)
# ═══════════════════════════════════════════════════════════════════════
def fig6_attack_intensity():
    time_s = np.arange(0, 30.1, 0.1)
    intensities = [0.3, 0.5, 0.8, 1.2, 2.0, 3.0]
    colors = [DEAKIN_BLUE, DARK_TEAL, "#7BAFD4", ORANGE, RED, "#800020"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for intensity, c in zip(intensities, colors):
        delta_phi = np.where(time_s >= 3.0, intensity, 0.1)
        gti = compute_gti(delta_phi, alpha=0.95, k=10.0, theta=2.0)
        ax1.plot(time_s, gti, color=c, linewidth=2,
                 label=f"ΔΦ = {intensity} m")

    ax1.axvspan(3.0, 30.0, alpha=0.05, color=RED)
    ax1.axhline(0.7, color=GRAY, linestyle=":", linewidth=1, alpha=0.4)
    ax1.axhline(0.3, color=GRAY, linestyle=":", linewidth=1, alpha=0.4)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("GPS Trust Index (GTI)")
    ax1.set_title("GTI Decay at Different Spoofing Intensities")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_xlim(0, 30)

    # Right: ΔΦ vs steady-state GTI + time-to-0.3
    deltas = np.linspace(0.05, 5.0, 100)
    t_long = np.linspace(0, 300, 3000)
    steady_gti = []
    t_to_03_list = []
    delta_phi_cont = np.full_like(t_long, 0.0)

    for d in deltas:
        delta_phi_cont[:] = d
        g = compute_gti(delta_phi_cont, alpha=0.95, k=10.0, theta=2.0)
        steady_gti.append(g[-1])
        idx = np.argmax(g <= 0.3)
        t_to_03_list.append(t_long[idx] if idx > 0 else 300)

    ax2.plot(deltas, steady_gti, color=DEAKIN_BLUE, linewidth=2.5,
             label="Steady-state GTI")
    ax2.set_xlabel("ΔΦ — GPS–IMU Inconsistency (m)")
    ax2.set_ylabel("Steady-State GTI", color=DEAKIN_BLUE)
    ax2.tick_params(axis="y", labelcolor=DEAKIN_BLUE)
    ax2.set_ylim(-0.02, 1.05)

    ax2b = ax2.twinx()
    ax2b.plot(deltas, t_to_03_list, color=RED, linewidth=2.5,
              label="Time to G ≤ 0.3")
    ax2b.set_ylabel("Time to G ≤ 0.3 (s)", color=RED)
    ax2b.tick_params(axis="y", labelcolor=RED)

    ax2.axvline(2.0 / 10.0, color=GRAY, linestyle=":", linewidth=1, alpha=0.6)
    ax2.text(0.22, 0.5, "θ/k = 0.2m", fontsize=8, color=GRAY, rotation=90,
             va="center")
    ax2.set_title("Steady-State GTI vs Attack Intensity")

    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, loc="center right", fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gti_attack_intensity.png")
    fig.savefig(path)
    print(f"  ✓ {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7 — Recovery after spoofing stops (hysteresis)
# ═══════════════════════════════════════════════════════════════════════
def fig7_recovery_hysteresis():
    time_s = np.arange(0, 30.1, 0.1)
    delta_phi = np.where((time_s >= 3.0) & (time_s < 18.0), 2.0, 0.1)

    gti_95 = compute_gti(delta_phi, alpha=0.95, k=10.0, theta=2.0)
    gti_90 = compute_gti(delta_phi, alpha=0.90, k=10.0, theta=2.0)
    gti_98 = compute_gti(delta_phi, alpha=0.98, k=10.0, theta=2.0)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.plot(time_s, gti_98, color=ORANGE, linewidth=2, label="α = 0.98 (slow recovery)")
    ax.plot(time_s, gti_95, color=DEAKIN_BLUE, linewidth=2.5, label="α = 0.95 (default)")
    ax.plot(time_s, gti_90, color=DARK_TEAL, linewidth=2, label="α = 0.90 (fast recovery)")

    ax.axvspan(3.0, 18.0, alpha=0.06, color=RED)
    ax.text(10.5, 0.95, "Spoofing Active\n(ΔΦ = 2.0m)", fontsize=10, color=RED,
            alpha=0.8, ha="center", style="italic")
    ax.text(24, 0.92, "Attack Stopped\nΔΦ -> 0.1m", fontsize=10, color=GREEN,
            alpha=0.8, ha="center", style="italic")

    ax.axhline(0.7, color=GRAY, linestyle=":", linewidth=1, alpha=0.4)
    ax.axhline(0.3, color=GRAY, linestyle=":", linewidth=1, alpha=0.4)

    # Mark recovery time
    for gti, a, c, y_off in [(gti_90, 0.90, DARK_TEAL, -0.03),
                               (gti_95, 0.95, DEAKIN_BLUE, 0.03),
                               (gti_98, 0.98, ORANGE, 0.0)]:
        idx = np.argmax((time_s > 18.0) & (gti > 0.7))
        if idx > 0:
            ax.axvline(time_s[idx], color=c, linestyle="--", linewidth=1, alpha=0.5)
            ax.text(time_s[idx] + 0.3, 0.7 + y_off, f"{time_s[idx]:.1f}s",
                    fontsize=8, color=c)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("GPS Trust Index (GTI)")
    ax.set_title("GTI Recovery After Spoofing Stops — Hysteresis Analysis\n(k=10, θ=2)")
    ax.legend(loc="lower left")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, 30)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gti_recovery_hysteresis.png")
    fig.savefig(path)
    print(f"  ✓ {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 8 — Full GTI worked example (thesis slide 15-16 style)
# ═══════════════════════════════════════════════════════════════════════
def fig8_worked_example():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left — Normal flight: ΔΦ ≈ 0.1m steady
    time_s = np.arange(0, 15.1, 0.1)
    delta_phi_normal = 0.1 * np.ones_like(time_s)
    gti_normal = compute_gti(delta_phi_normal, alpha=0.95, k=10.0, theta=2.0, g0=1.0)

    ax1.plot(time_s, gti_normal, color=GREEN, linewidth=3)
    ax1.fill_between(time_s, 0.9, 1.0, alpha=0.1, color=GREEN)
    ax1.axhline(0.987, linestyle="--", color=GREEN, linewidth=1, alpha=0.5)
    ax1.text(12, 0.99, "G_t ≈ 0.987", fontsize=12, color=GREEN, fontweight="bold",
             ha="right")
    ax1.set_ylim(0.95, 1.02)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("GPS Trust Index (GTI)")
    axis_title = ("Normal Flight — Trust Remains Stable\n"
                  "ΔΦ = 0.1m (GPS & IMU agree)")
    ax1.set_title(axis_title, color=GREEN, fontweight="bold")

    # Right — Spoofing: ΔΦ = 0.8m sustained
    time_s2 = np.arange(0, 20.1, 0.05)
    delta_phi_spoof = np.where(time_s2 >= 0.0, 0.8, 0.1)
    gti_spoof = compute_gti(delta_phi_spoof, alpha=0.95, k=10.0, theta=2.0, g0=1.0)

    milestones = [0.7, 0.3, 0.1, 0.05]
    milestone_labels = ["Attenuate GPS Weight", "Re-weight Sensors",
                        "HOLD Mode", "Emergency RTL"]
    milestone_colors = [ORANGE, "#E6A817", RED, "#800020"]

    ax2.plot(time_s2, gti_spoof, color=RED, linewidth=3, zorder=10)

    last_t = 0
    for i, (m, label, c) in enumerate(zip(milestones, milestone_labels, milestone_colors)):
        idx = np.argmax(gti_spoof <= m)
        t_m = time_s2[idx] if idx > 0 else 20
        ax2.axhline(m, color=c, linestyle=":", linewidth=1.2, alpha=0.7, zorder=4)
        ax2.axvline(t_m, color=c, linestyle=":", linewidth=1.2, alpha=0.5, zorder=4)
        ax2.scatter([t_m], [m], color=c, s=50, zorder=20, edgecolors="white",
                    linewidth=0.8)
        ax2.annotate(f"t={t_m:.1f}s\n{label}", xy=(t_m, m),
                     xytext=(t_m + 0.8, m + 0.06), fontsize=8, color=c,
                     fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
        last_t = t_m

    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlim(0, 20)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("GPS Trust Index (GTI)")
    axis_title2 = ("Under Spoofing — Trust Decays -> Escalation\n"
                   "ΔΦ = 0.8m, α=0.95, k=10, θ=2")
    ax2.set_title(axis_title2, color=RED, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gti_worked_example.png")
    fig.savefig(path)
    print(f"  ✓ {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 9 — Effect of k on sensitivity with GTI threshold lines
# ═══════════════════════════════════════════════════════════════════════
def fig9_k_sensitivity_curves():
    time_s = np.arange(0, 20.1, 0.1)
    delta_phi = np.where(time_s >= 2.0, 0.5, 0.1)

    ks = [3, 5, 10, 20, 40]
    colors = [DEAKIN_BLUE, DARK_TEAL, GREEN, ORANGE, RED]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for k, c in zip(ks, colors):
        gti = compute_gti(delta_phi, alpha=0.95, k=k, theta=2.0)
        ax.plot(time_s, gti, color=c, linewidth=2, label=f"k = {k}")

    ax.axvspan(2.0, 20.0, alpha=0.05, color=RED)
    ax.text(11, 0.97, "Spoofing Active (ΔΦ = 0.5m)", fontsize=9, color=RED,
            alpha=0.7, ha="center", style="italic")

    ax.axhline(0.7, color=GRAY, linestyle=":", linewidth=1, alpha=0.4)
    ax.axhline(0.3, color=GRAY, linestyle=":", linewidth=1, alpha=0.4)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("GPS Trust Index (GTI)")
    ax.set_title("Effect of Scaling Factor k on Detection Sensitivity\n(α=0.95, θ=2, ΔΦ=0.5m)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, 20)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "gti_k_sensitivity.png")
    fig.savefig(path)
    print(f"  ✓ {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 10 — Summary: All GTI parameters impact
# ═══════════════════════════════════════════════════════════════════════
def fig10_summary_dashboard():
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    time_s = np.arange(0, 25.1, 0.1)
    delta_phi_spoof = np.where(time_s >= 3.0, 1.5, 0.1)

    # (0,0) — α sweep
    ax = fig.add_subplot(gs[0, 0])
    for a, c in zip([0.85, 0.90, 0.95, 0.98],
                    [ORANGE, RED, DEAKIN_BLUE, DARK_TEAL]):
        gti = compute_gti(delta_phi_spoof, alpha=a, k=10.0, theta=2.0)
        ax.plot(time_s, gti, color=c, linewidth=2, label=f"α={a}")
    ax.axvspan(3.0, 25.0, alpha=0.05, color=RED)
    ax.axhline(0.3, color=GRAY, linestyle=":", linewidth=1, alpha=0.4)
    ax.set_title("Smoothing α\n(Longer memory = slower decay)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("GTI")

    # (0,1) — k sweep
    ax = fig.add_subplot(gs[0, 1])
    for k, c in zip([3, 5, 10, 20], [DEAKIN_BLUE, DARK_TEAL, GREEN, RED]):
        gti = compute_gti(delta_phi_spoof, alpha=0.95, k=k, theta=2.0)
        ax.plot(time_s, gti, color=c, linewidth=2, label=f"k={k}")
    ax.axvspan(3.0, 25.0, alpha=0.05, color=RED)
    ax.axhline(0.3, color=GRAY, linestyle=":", linewidth=1, alpha=0.4)
    ax.set_title("Scaling k\n(Higher = more sensitive)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(-0.02, 1.05)

    # (0,2) — θ sweep
    ax = fig.add_subplot(gs[0, 2])
    for th, c in zip([0.5, 1.0, 2.0, 4.0], [RED, ORANGE, DEAKIN_BLUE, DARK_TEAL]):
        gti = compute_gti(delta_phi_spoof, alpha=0.95, k=10.0, theta=th)
        ax.plot(time_s, gti, color=c, linewidth=2, label=f"θ={th}")
    ax.axvspan(3.0, 25.0, alpha=0.05, color=RED)
    ax.axhline(0.3, color=GRAY, linestyle=":", linewidth=1, alpha=0.4)
    ax.set_title("Offset θ\n(Higher = more tolerant)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_ylim(-0.02, 1.05)

    # (1, 0:3) — Graduated response full width
    ax = fig.add_subplot(gs[1, :])
    time_long = np.arange(0, 25.1, 0.05)
    delta_long = np.where(time_long >= 2.0, 1.8, 0.1)
    gti = compute_gti(delta_long, alpha=0.95, k=10.0, theta=2.0)
    ax.plot(time_long, gti, color=DEAKIN_BLUE, linewidth=3)
    zones_data = [
        (0.7, 1.0, GREEN, "Full Trust"),
        (0.3, 0.7, ORANGE, "Attenuate GPS"),
        (0.1, 0.3, "#E6A817", "Re-weight"),
        (0.05, 0.1, RED, "HOLD"),
        (0.0, 0.05, "#800020", "RTL"),
    ]
    for lo, hi, col, lab in zones_data:
        ax.fill_between(time_long, lo, hi, alpha=0.07, color=col)
    ax.set_title("Graduated Response Escalation (α=0.95, k=10, θ=2, ΔΦ=1.8m)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("GTI")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, 25)

    # (2, 0) — Normal vs spoof GTI
    ax = fig.add_subplot(gs[2, 0])
    t_ns = np.arange(0, 20.1, 0.2)
    np.random.seed(123)
    dp_n = 0.1 + 0.06 * np.abs(np.random.randn(len(t_ns)))
    dp_s = np.where(t_ns >= 4.0, 1.2, dp_n)
    g_n = compute_gti(dp_n, alpha=0.95, k=10.0, theta=2.0)
    g_s = compute_gti(dp_s, alpha=0.95, k=10.0, theta=2.0)
    ax.plot(t_ns, g_n, color=GREEN, linewidth=2, label="Normal")
    ax.plot(t_ns, g_s, color=RED, linewidth=2, label="Spoofed")
    ax.axvspan(4.0, 20.0, alpha=0.05, color=RED)
    ax.axhline(0.3, color=GRAY, linestyle=":", linewidth=1, alpha=0.4)
    ax.set_title("Normal vs Spoofing", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("GTI")

    # (2, 1) — Intensity sensitivity
    ax = fig.add_subplot(gs[2, 1])
    for intens, c in zip([0.3, 0.8, 2.0], [DEAKIN_BLUE, ORANGE, RED]):
        dp = np.where(t_ns >= 4.0, intens, 0.1)
        g = compute_gti(dp, alpha=0.95, k=10.0, theta=2.0)
        ax.plot(t_ns, g, color=c, linewidth=2, label=f"ΔΦ={intens}m")
    ax.axvspan(4.0, 20.0, alpha=0.05, color=RED)
    ax.axhline(0.3, color=GRAY, linestyle=":", linewidth=1, alpha=0.4)
    ax.set_title("Attack Intensity", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(-0.02, 1.05)

    # (2, 2) — Recovery / hysteresis
    ax = fig.add_subplot(gs[2, 2])
    dp_rec = np.where((t_ns >= 4.0) & (t_ns < 12.0), 2.0, 0.1)
    g_095 = compute_gti(dp_rec, alpha=0.95, k=10.0, theta=2.0)
    g_090 = compute_gti(dp_rec, alpha=0.90, k=10.0, theta=2.0)
    ax.plot(t_ns, g_090, color=DARK_TEAL, linewidth=2, label="α=0.90")
    ax.plot(t_ns, g_095, color=DEAKIN_BLUE, linewidth=2.5, label="α=0.95")
    ax.axvspan(4.0, 12.0, alpha=0.05, color=RED)
    ax.set_title("Recovery After Attack", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Time (s)")

    fig.suptitle("GPS Trust Index (GTI) — Parameter Analysis Dashboard\n"
                 "$G_t = \\alpha \\, G_{t-1} + (1-\\alpha) \\, "
                 "(1 - \\sigma(k \\cdot \\Delta\\Phi_t - \\theta))$",
                 fontsize=13, fontweight="bold", y=1.01)

    path = os.path.join(OUT_DIR, "gti_summary_dashboard.png")
    fig.savefig(path)
    print(f"  ✓ {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60)
    print(" Generating GTI Parameter Analysis Graphs")
    print("=" * 60)
    os.makedirs(OUT_DIR, exist_ok=True)

    fig1_sigmoid_components()
    fig2_alpha_sensitivity()
    fig3_normal_vs_spoof()
    fig4_graduated_response()
    fig5_parameter_heatmaps()
    fig6_attack_intensity()
    fig7_recovery_hysteresis()
    fig8_worked_example()
    fig9_k_sensitivity_curves()
    fig10_summary_dashboard()

    print("\n" + "=" * 60)
    print(f" Done! All graphs saved to:\n   {OUT_DIR}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
