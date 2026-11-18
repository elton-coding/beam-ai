"""Receiver-centered propagation heatmaps for focus and current direct NN."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from direct_network.core.physics import AngularSpectrum, Scene, parameter_bounds
from direct_network.core.paths import ROOT




def profile(engine: AngularSpectrum, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scene = engine.scene
    za = scene.obstacle_z_m - scene.obstacle_depth_m / 2
    zb = scene.obstacle_z_m + scene.obstacle_depth_m / 2
    pre = np.linspace(0.0, za, 94)
    inside = np.linspace(za, zb, engine.n_steps + 1)[1:]
    post = np.linspace(zb, scene.receiver_z_m, 74)[1:]
    z_values = np.r_[pre, inside, post]
    field = source.copy()
    fields = [field.copy()]
    previous = 0.0
    for z in z_values[1:]:
        dz = z - previous
        transfer = np.exp(1j * engine.kz * dz)
        field = engine._propagate(field, transfer)
        if za - 1e-12 <= z <= zb + 1e-12:
            field *= engine.block_mask
        fields.append(field.copy())
        previous = z
    expected = engine.received_field(source)
    relative_error = np.linalg.norm(fields[-1] - expected) / np.linalg.norm(expected)
    if relative_error > 1e-9:
        raise AssertionError(f"Heatmap propagation disagrees with power model: {relative_error}")
    return z_values, np.abs(np.asarray(fields)) ** 2


def main() -> None:
    scene = Scene(n_grid=16384, obstacle_step_m=0.00075)
    engine = AngularSpectrum(scene)
    x_m = 0.015
    table = pd.read_csv(ROOT / "results" / "static_results.csv")
    row = table[np.isclose(table.receiver_x_m, x_m, atol=1e-12)].iloc[0]
    _, _, f0, theta0 = parameter_bounds(scene, x_m)
    sources = [engine.source(0.0, f0, theta0),
               engine.source(row.B_per_m, row.F_m, row.theta_deg)]
    profiles = [profile(engine, source) for source in sources]
    z = profiles[0][0]
    mask = np.abs(engine.x) <= 0.11
    maximum = max(float(np.max(p[1][:, mask])) for p in profiles)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharey=True, constrained_layout=True)
    labels = ["Focused beam", "Current direct NN"]
    for ax, (zs, values), title in zip(axes, profiles, labels):
        db = 10 * np.log10(np.maximum(values[:, mask] / maximum, 1e-8))
        mesh = ax.pcolormesh(engine.x[mask] * 1000, zs * 1000,
                             np.clip(db, -35, 0), cmap="magma", vmin=-35, vmax=0,
                             shading="nearest", rasterized=True)
        ax.add_patch(Rectangle(((scene.obstacle_x_m - scene.obstacle_width_m/2)*1000,
                                (scene.obstacle_z_m - scene.obstacle_depth_m/2)*1000),
                               scene.obstacle_width_m*1000, scene.obstacle_depth_m*1000,
                               facecolor="black", edgecolor="white", lw=0.7))
        rx_left = (x_m - scene.rx_width_m / 2) * 1000
        rx_right = (x_m + scene.rx_width_m / 2) * 1000
        rx_z = scene.receiver_z_m * 1000
        ax.axhline(rx_z, color="#24C5AE", ls="--", lw=1, alpha=0.8)
        ax.plot([rx_left, rx_right], [rx_z, rx_z], color="black", lw=11,
                solid_capstyle="butt", zorder=8)
        ax.plot([rx_left, rx_right], [rx_z, rx_z], color="#24C5AE", lw=7,
                solid_capstyle="butt", zorder=9)
        ax.annotate("RX aperture\nx=15 mm, z=360 mm",
                    xy=(x_m * 1000, rx_z), xytext=(67, 321),
                    color="#07564F", fontsize=8, ha="center", va="center",
                    bbox={"facecolor": "white", "edgecolor": "#24C5AE", "alpha": 0.94, "pad": 2},
                    arrowprops={"arrowstyle": "->", "color": "#24C5AE", "lw": 1.6},
                    zorder=10)
        ax.set(title=title, xlabel="Transverse position x (mm)", xlim=(-110, 110), ylim=(392, 0))
    axes[0].set_ylabel("Propagation distance z (mm)")
    colorbar = fig.colorbar(mesh, ax=axes, shrink=0.94, pad=0.02)
    colorbar.set_label("Intensity relative to shared peak (dB)")
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "propagation_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
