"""Figures for the outside-reader technical report.

Run: python -m nfbeam.report_figures [--skip-fullwave]

Writes to ``nfbeam/figures/report/``:
- copies of the E1-E11 result figures with reader-facing labels
  ("学习型基线" / "本文方法" instead of the internal "原方案" / "新方案");
- beam-propagation heatmaps |E(x, z)|^2 for one deep-shadow position:
  angular-spectrum model (focus / best Airy / per-element phase) and
  full-wave solution (metal TM / TE).
"""

from __future__ import annotations

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from direct_network.core.physics import blockage_ratio
from nfbeam import design as D
from nfbeam import plots
from nfbeam.channel import Channel
from nfbeam.scenes import FIGURES, load_scenes

OUT = FIGURES / "report"

# Longest phrases first: the general words would otherwise eat the specific ones.
RELABEL = [
    ("原方案：学习型 Airy 网络", "学习型 Airy（基线）"),
    ("原方案网络", "学习型基线"),
    ("新方案（实时）深遮挡", "本文方法（实时版）深遮挡"),
    ("新方案（实时版）", "本文方法（实时版）"),
    ("新方案（实时）", "本文方法（实时版）"),
    ("新方案（不处理时延）", "本文方法（不处理时延）"),
    ("新方案（时延感知）", "本文方法（时延感知）"),
    ("新跟踪 + 聚焦", "贝叶斯跟踪 + 聚焦"),
    ("原方案", "学习型基线"),
    ("新方案", "本文方法"),
    ("码本波束训练", "码本训练"),
]


def _relabel(s: str) -> str:
    for old, new in RELABEL:
        s = s.replace(old, new)
    return re.sub(r"^E\d+b? ", "", s)


def _save_relabelled(fig, name: str) -> None:
    for t in fig.findobj(Text):
        if t.get_text():
            t.set_text(_relabel(t.get_text()))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def result_figures() -> None:
    plots.save = _save_relabelled
    plots.main()


# ---------------------------------------------------------------- heatmaps
X_HALF = 0.15
Z_MAX = 0.42


def pick_position(scene) -> float:
    """Middle of the deep-shadow range: a typical position, not a chosen one."""
    grid = np.linspace(-0.11, 0.11, 881)
    deep = grid[[blockage_ratio(scene, float(x)) >= scene.activation_bl for x in grid]]
    return float(deep[np.argmin(np.abs(deep - deep.mean()))])


def as_field_map(ch: Channel, w: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """|E|^2 on (z, x) from the angular-spectrum model with the obstacle masks."""
    e, sc = ch.engine, ch.scene
    src = np.zeros(sc.n_grid, dtype=complex)
    src[ch._basis_cols] = np.asarray(w) @ ch._basis
    za = sc.obstacle_z_m - sc.obstacle_depth_m / 2
    zb = sc.obstacle_z_m + sc.obstacle_depth_m / 2
    prop = lambda f, dz: Channel._propagate(f, np.exp(1j * e.kz * dz))
    # Field just behind each absorbing slice, to look up inside the obstacle.
    f = prop(src, za) * e.block_mask
    slices = [(za, f)]
    for i in range(e.n_steps):
        f = prop(f, e.actual_step) * e.block_mask
        slices.append((za + (i + 1) * e.actual_step, f))
    f_zb = f
    keep = np.abs(e.x) <= X_HALF
    out = np.empty((len(zs), keep.sum()))
    for i, z in enumerate(zs):
        if z < za:
            g = prop(src, z)
        elif z <= zb:
            z0, g0 = max((s for s in slices if s[0] <= z), key=lambda s: s[0])
            g = prop(g0, z - z0)
        else:
            g = prop(f_zb, z - zb)
        out[i] = np.abs(g[keep]) ** 2
    return out, e.x[keep]


def _draw(ax, img, xs, zs, scene, x_rx, title, ref, vmin=-25, vmax=10):
    db = 10 * np.log10(np.maximum(img / ref, 1e-12))
    im = ax.imshow(db.T, origin="lower", aspect="auto", cmap="inferno", vmin=vmin, vmax=vmax,
                   extent=[zs[0] * 100, zs[-1] * 100, xs[0] * 100, xs[-1] * 100])
    za = scene.obstacle_z_m - scene.obstacle_depth_m / 2
    ax.add_patch(Rectangle((za * 100, (scene.obstacle_x_m - scene.obstacle_width_m / 2) * 100),
                           scene.obstacle_depth_m * 100, scene.obstacle_width_m * 100,
                           facecolor="#3a3a3a", edgecolor="white", lw=1.2))
    ax.plot([0, 0], [-scene.tx_width_m * 50, scene.tx_width_m * 50], color="#7fd3ff", lw=4)
    ax.plot([scene.receiver_z_m * 100] * 2,
            [(x_rx - scene.rx_width_m / 2) * 100, (x_rx + scene.rx_width_m / 2) * 100],
            color="#7fd3ff", lw=4)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("传播距离 z (cm)")
    ax.grid(False)
    return im


def heatmaps(skip_fullwave: bool) -> None:
    sid, scene = load_scenes("test")[0]
    scene = scene  # first held-out scene
    ch = Channel(scene)
    x = pick_position(scene)
    A = ch.gram_at(x)
    beams = {"focus": D.focus_beam(ch, x), "airy": D.airy_search(ch, A, x), "phase": D.best_phase_only(ch, A, x)}
    p = {k: ch.power(w, x) for k, w in beams.items()}
    zs = np.linspace(0.002, Z_MAX, 420)
    maps = {k: as_field_map(ch, w, zs) for k, w in beams.items()}
    xs = maps["focus"][1]
    ref = float(np.mean(maps["focus"][0][0][np.abs(xs) < scene.tx_width_m / 2 - 0.01]))
    names = {"focus": "聚焦波束", "airy": "最优 Airy 波束", "phase": "逐阵元纯相位"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, k in zip(axes, beams):
        gain = 10 * np.log10(p[k] / p["focus"])
        sub = "" if k == "focus" else f"，接收功率 {gain:+.1f} dB"
        im = _draw(ax, maps[k][0], xs, zs, scene, x, names[k] + sub, ref)
    axes[0].set_ylabel("横向位置 x (cm)")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
    cb.set_label("强度 (dB，相对孔径处平均)")
    fig.suptitle(f"波束传播热力图（角谱模型）：测试场景 {sid}，接收机 x = {100 * x:.1f} cm，"
                 f"遮挡率 {100 * blockage_ratio(scene, x):.0f}%",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "beam_heatmap_as.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("heatmap as", sid, round(x, 4), {k: round(10 * np.log10(v / p["focus"]), 2) for k, v in p.items()})
    if not skip_fullwave:
        fullwave_heatmaps(scene, sid, x, beams, ref)


def fullwave_field(scene, bc: str, beams: list[np.ndarray], obs: np.ndarray, chunk: int = 3000) -> np.ndarray:
    from nfbeam.fullwave import Scatterer, aperture_field, rectangle
    ch = Channel(scene)
    verts = rectangle(scene.obstacle_x_m, scene.obstacle_z_m, scene.obstacle_width_m, scene.obstacle_depth_m)
    sc = Scatterer(verts, ch.k, bc, 10.0)
    W = np.stack(beams, axis=1)                                     # (n_tx, n_beams)
    u_bnd = aperture_field(sc.rhs_points(), ch.centers, ch.pitch, ch.k).T @ W
    density = sc.solve(u_bnd)
    out = np.empty((len(obs), len(beams)), dtype=complex)
    for lo in range(0, len(obs), chunk):
        o = obs[lo:lo + chunk]
        out[lo:lo + chunk] = aperture_field(o, ch.centers, ch.pitch, ch.k).T @ W + sc.scattered(o, density)
    return out


def fullwave_heatmaps(scene, sid, x, beams, ref) -> None:
    xs = np.arange(-X_HALF, X_HALF + 1e-9, 0.001)
    zs = np.arange(0.004, Z_MAX + 1e-9, 0.0015)
    Z, X = np.meshgrid(zs, xs, indexing="ij")
    obs = np.stack([X.ravel(), Z.ravel()], axis=1)
    za, zb = scene.obstacle_z_m - scene.obstacle_depth_m / 2, scene.obstacle_z_m + scene.obstacle_depth_m / 2
    inside = (np.abs(obs[:, 0] - scene.obstacle_x_m) <= scene.obstacle_width_m / 2) & \
             (obs[:, 1] >= za) & (obs[:, 1] <= zb)
    panels = []
    for bc in ("tm", "te"):
        u = fullwave_field(scene, bc, [beams["focus"], beams["phase"]], obs[~inside])
        for j, k in enumerate(("focus", "phase")):
            img = np.zeros(len(obs))
            img[~inside] = np.abs(u[:, j]) ** 2
            panels.append((bc, k, img.reshape(Z.shape)))
        print("fullwave", bc)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    label = {"tm": "金属 TM", "te": "金属 TE"}
    names = {"focus": "聚焦波束", "phase": "逐阵元纯相位（按角谱模型设计）"}
    for ax, (bc, k, img) in zip(axes.ravel(), panels):
        im = _draw(ax, img, xs, zs, scene, x, f"全波 · {label[bc]} · {names[k]}", ref)
    for ax in axes[:, 0]:
        ax.set_ylabel("横向位置 x (cm)")
    for ax in axes[0]:
        ax.set_xlabel("")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("强度 (dB，相对孔径处平均)")
    fig.suptitle(f"波束传播热力图（二维全波解）：同一场景 {sid}、同一接收位置", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.savefig(OUT / "beam_heatmap_fullwave.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fullwave", action="store_true")
    ap.add_argument("--heatmaps-only", action="store_true")
    a = ap.parse_args()
    if not a.heatmaps_only:
        result_figures()
    heatmaps(a.skip_fullwave)


if __name__ == "__main__":
    main()
