"""Frame-level simulation: sensing, feedback (delay/loss), beam design, payload.

One scene context holds the nominal geometry hypotheses, the truth channel and
the precomputed scan codebooks.  Every policy in one run sees the same
trajectory, gains, multipath phases, noise and feedback-loss pattern, so the
comparisons are paired.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from direct_network.core.physics import blockage_ratio
from nfbeam import design as D
from nfbeam.channel import Channel
from nfbeam.scenes import hypotheses, random_mismatch
from nfbeam.sensing import (N_SUB, PILOT_S, EchoConfig, EchoModel, Impairments, combine_hypotheses,
                            complex_normal, coop_loglik, coop_observe, coop_template, echo_template,
                            payload_power, update_geometry)
from nfbeam.tracking import X_GRID, DelayedFilter, FilterConfig, GridFilter


FRAME_S = 0.02
TAU = PILOT_S
FEEDBACK_S = 1e-4
SCAN_X = np.linspace(-0.10, 0.10, 9)
CODEBOOK_X = np.linspace(-0.10, 0.10, 17)
PASSIVE_LEGACY_X = np.linspace(-0.10, 0.10, 21)
RESCAN_STD_M = 0.005
LOCK_ERROR_M = 0.010
RHO_DB = np.array([-8.0, -6.0, -4.0, -2.0, 0.0, 2.0])


@dataclass(frozen=True)
class Condition:
    name: str = "matched"
    geometry: str = "matched"        # matched | mismatch | strong
    pilot_noise: float = 3e-4
    delay: int = 0                   # feedback delay in frames
    loss: str = "none"               # none | iid10 | iid30 | burst
    echo: EchoConfig = field(default_factory=EchoConfig)


def trajectory(run: int) -> np.ndarray:
    """Same five 101-frame crossings as the previous pilot-reuse experiment."""
    t = np.linspace(0, 1, 101)
    if run < 3:
        t = t + .09 * np.sin(2 * np.pi * t + run) - .09 * np.sin(run)
    else:
        t = t + .11 * np.sin(4 * np.pi * t)
    return (-1 if run % 2 else 1) * (.19 * t - .095)


def make_losses(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Feedback-loss pattern: none, i.i.d. (``iid10`` = 10 %), or Gilbert-Elliott bursts."""
    if kind == "none":
        return np.zeros(n, bool)
    if kind.startswith("iid"):
        return rng.random(n) < int(kind[3:]) / 100
    if kind == "burst":   # good->bad 3 %, bad->good 25 %: ~11 % of frames bad, mean burst 4 frames
        out, bad = np.zeros(n, bool), False
        for i in range(n):
            bad = (rng.random() >= 0.25) if bad else (rng.random() < 0.03)
            out[i] = rng.random() < (0.9 if bad else 0.01)
        return out
    raise ValueError(kind)


# ------------------------------------------------------------------ context
class SceneContext:
    def __init__(self, scene_id: int, scene, cond: Condition, seed: int, need_legacy: bool = True,
                 need_codebook: bool = False):
        rng = np.random.default_rng(seed)
        self.scene_id, self.scene, self.cond = scene_id, scene, cond
        self.imp = Impairments(pilot_noise=cond.pilot_noise)
        self.hyps = [Channel(s) for s in hypotheses(scene)]
        self.nominal = self.hyps[len(self.hyps) // 2]
        if cond.geometry == "matched":
            self.truth = self.nominal
        else:
            self.truth = Channel(random_mismatch(scene, rng, strong=cond.geometry == "strong"))
        pi = np.full(len(self.hyps), 0.5 / (len(self.hyps) - 1))
        pi[len(self.hyps) // 2] = 0.5
        self.log_pi0 = np.log(pi)
        ref = D.focus_beam(self.nominal, 0.095)
        self.data_noise = self.truth.power(ref, 0.095) / 10
        self.scan_beams = [self._shadow_beam(x, 0.0125) for x in SCAN_X]
        self.scan_templates = [np.vstack([coop_template(h, w) for w in self.scan_beams]) for h in self.hyps]
        if need_codebook:
            self.codebook = [self._shadow_beam(x, 0.006) for x in CODEBOOK_X]
        if need_legacy:
            self._build_legacy()

    def _shadow_beam(self, x: float, sigma: float) -> np.ndarray:
        px = np.exp(-0.5 * ((X_GRID - x) / sigma) ** 2)
        A = D.robust_gram([self.nominal], np.array([1.0]), X_GRID, px / px.sum())
        return D.best_phase_only(self.nominal, A, x)

    def _build_legacy(self) -> None:
        from direct_network.core.general_model import GeneralSceneNetwork
        from direct_network.experiments.evaluate_general_scene import direct_parameters
        from direct_network.experiments.evaluate_pilot_reuse import GRID_X
        net = GeneralSceneNetwork()
        self.legacy_beams = [D.airy_beam(self.nominal, *direct_parameters(self.scene, net, float(g)))
                             for g in GRID_X]
        amp = lambda w: np.sqrt(np.maximum(self.nominal.aperture_power(w, GRID_X, N_SUB), 0))
        self.legacy_templates = [amp(w) for w in self.legacy_beams]
        self.legacy_scan_beams = [D.focus_beam(self.nominal, x) for x in SCAN_X]
        self.legacy_scan_template = np.vstack([amp(w) for w in self.legacy_scan_beams])

    def templates(self, w: np.ndarray) -> list[np.ndarray]:
        return [coop_template(h, w) for h in self.hyps]

    def deep(self, x: float) -> bool:
        return blockage_ratio(self.truth.scene, float(x)) >= self.scene.activation_bl


# ------------------------------------------------------------------- design
def design_beam(ctx: SceneContext, px: np.ndarray, log_pi: np.ndarray, family: str,
                prev: np.ndarray | None) -> tuple[np.ndarray, float]:
    keep = px > 1e-4 * px.max()
    xs, ps = X_GRID[keep], px[keep] / px[keep].sum()
    x_ref = float(ps @ xs)
    A = D.robust_gram(ctx.hyps, np.exp(log_pi), xs, ps)
    w_focus = D.focus_beam(ctx.nominal, x_ref)
    if family == "focus":
        return w_focus, x_ref
    if family == "airy":
        w = D.airy_search(ctx.nominal, A, x_ref, max_eval=120)
    else:
        w = D.best_phase_only(ctx.nominal, A, x_ref, extra_seeds=[prev] if prev is not None else None)
    return (w if D.quad(A, w) >= D.quad(A, w_focus) else w_focus), x_ref


@dataclass
class RunRandom:
    gains: np.ndarray
    phases: np.ndarray
    noise: np.ndarray          # (frames, 40, N_SUB) complex, sliced per use
    lost: np.ndarray           # (frames,) feedback lost
    lost_scan: np.ndarray

    @classmethod
    def draw(cls, n: int, seed: int, loss: str) -> "RunRandom":
        rng = np.random.default_rng(seed)
        return cls(10 ** (rng.normal(0, .75, n) / 10), rng.uniform(-np.pi, np.pi, n),
                   complex_normal(rng, (n, 40, N_SUB)), make_losses(loss, n, rng), make_losses(loss, n, rng))


# --------------------------------------------------------- cooperative runs
def run_coop(ctx: SceneContext, policy: str, path: np.ndarray, rr: RunRandom,
             delay_aware: bool = True) -> list[dict]:
    if policy == "legacy":
        return _run_legacy(ctx, path, rr)
    if policy == "oracle":
        return _run_oracle(ctx, path, rr)
    if policy == "codebook":
        return _run_codebook(ctx, path, rr)
    family = {"focus_bayes": "focus", "proposed": "phase", "proposed_verified": "phase",
              "proposed_airy": "airy"}[policy]
    verify = policy == "proposed_verified"
    cond, imp = ctx.cond, ctx.imp
    filt = DelayedFilter(GridFilter(FilterConfig()), aware=delay_aware)
    log_pi = ctx.log_pi0.copy()
    queue: list[tuple[int, int, list[np.ndarray], np.ndarray]] = []
    prev = None
    rows = []

    def apply(frame: int, templates: list[np.ndarray], y: np.ndarray) -> None:
        nonlocal log_pi
        ll_h = coop_loglik(templates, y, imp, imp.pilot_noise)
        log_pi = update_geometry(log_pi, ll_h, filt.filt.marginal_x())
        filt.add(frame, combine_hypotheses(ll_h, log_pi))

    for t, x in enumerate(path):
        filt.advance()
        for item in [q for q in queue if q[0] < t]:
            queue.remove(item)
            apply(item[1], item[2], item[3])
        mu, sd = filt.filt.mean_std()
        scanned = t == 0 or sd > RESCAN_STD_M
        overhead = TAU + FEEDBACK_S
        g, ph, nz = rr.gains[t], rr.phases[t], rr.noise[t]
        if scanned:
            overhead += len(SCAN_X) * TAU
            y = np.concatenate([coop_observe(ctx.truth, w, x, g, ph, nz[i], imp)
                                for i, w in enumerate(ctx.scan_beams)])
            if not rr.lost_scan[t]:
                if cond.delay == 0:
                    apply(t, ctx.scan_templates, y)
                else:
                    queue.append((t + cond.delay, t, ctx.scan_templates, y))
        px = filt.filt.marginal_x()
        est, est_sd = filt.filt.mean_std()
        w, x_ref = design_beam(ctx, px, log_pi, family, prev)
        if verify and cond.delay == 0:
            overhead += 2 * TAU
            cands = [w, D.focus_beam(ctx.nominal, x_ref)]
            ys = [coop_observe(ctx.truth, c, x, g, ph, nz[20 + i], imp) for i, c in enumerate(cands)]
            if not rr.lost[t]:
                w = cands[int(np.argmax([np.sum(v**2) for v in ys]))]
                for c, v in zip(cands, ys):
                    apply(t, ctx.templates(c), v)
        prev = w
        power = payload_power(ctx.truth, w, x, g, ph, imp)
        y_ref = coop_observe(ctx.truth, w, x, g, ph, nz[30], imp)
        if not rr.lost[t]:
            queue.append((t + cond.delay, t, ctx.templates(w), y_ref))
        rows.append(_row(ctx, t, x, est, est_sd, scanned, overhead, power))
    return rows


def _row(ctx, t, x, est, sd, scanned, overhead, power) -> dict:
    frac = (FRAME_S - overhead) / FRAME_S
    return {"frame": t, "actual_x_m": float(x), "estimate_x_m": est, "sigma_m": sd,
            "deep": ctx.deep(x), "scanned": bool(scanned), "payload_fraction": frac,
            "power": power, "se_bpshz": frac * np.log2(1 + power / ctx.data_noise)}


def _run_oracle(ctx, path, rr) -> list[dict]:
    rows, prev = [], None
    for t, x in enumerate(path):
        A = ctx.truth.gram_at(x)
        w = D.best_phase_only(ctx.truth, A, x, extra_seeds=[prev] if prev is not None else None)
        prev = w
        power = payload_power(ctx.truth, w, x, rr.gains[t], rr.phases[t], ctx.imp)
        rows.append(_row(ctx, t, x, float(x), 0.0, False, TAU + FEEDBACK_S, power))
    return rows


def _run_codebook(ctx, path, rr) -> list[dict]:
    """Classical beam training: probe a 17-beam codebook every frame, keep the best."""
    rows, best = [], len(ctx.codebook) // 2
    for t, x in enumerate(path):
        g, ph, nz = rr.gains[t], rr.phases[t], rr.noise[t]
        y = [np.sum(coop_observe(ctx.truth, w, x, g, ph, nz[i % 40], ctx.imp) ** 2)
             for i, w in enumerate(ctx.codebook)]
        if not rr.lost[t] and ctx.cond.delay == 0:
            best = int(np.argmax(y))
        w = ctx.codebook[best]
        power = payload_power(ctx.truth, w, x, g, ph, ctx.imp)
        overhead = TAU + FEEDBACK_S + len(ctx.codebook) * TAU
        rows.append(_row(ctx, t, x, float(CODEBOOK_X[best]), np.nan, True, overhead, power))
    return rows


def _run_legacy(ctx, path, rr) -> list[dict]:
    """Previous scheme: Kalman + template gate + recovery sweep + learned Airy net.

    Delayed reports are applied as if they were current (no out-of-sequence logic).
    """
    from direct_network.core.sensing import CausalPositionTracker
    from direct_network.experiments.evaluate_pilot_reuse import GRID_X, design_index, match
    cond, imp = ctx.cond, ctx.imp
    tracker = CausalPositionTracker(dt_s=FRAME_S, acceleration_std_m_s2=.5)
    misses, rows, queue = 0, [], []
    d = cond.delay
    for t, x in enumerate(path):
        g, ph, nz = rr.gains[t], rr.phases[t], rr.noise[t]
        scanned = t == 0 or misses >= 2
        prior = float(np.clip(tracker.state[0] + FRAME_S * tracker.state[1], GRID_X[0], GRID_X[-1]))
        arrived = [q for q in queue if q[0] == t]
        for q in arrived:
            queue.remove(q)
        late = next((q[1] for q in reversed(arrived) if q[1] is not None), None)
        overhead = TAU + FEEDBACK_S
        if scanned:
            overhead += len(SCAN_X) * TAU
            amp = np.concatenate([coop_observe(ctx.truth, w, x, g, ph, nz[i], imp)
                                  for i, w in enumerate(ctx.legacy_scan_beams)])
            m, _ = match(amp, ctx.legacy_scan_template, imp.pilot_noise)
            m = None if rr.lost_scan[t] else m
            if d == 0:                    # original timing: sweep result used in-frame
                estimate, sigma = tracker.step(m)
                beam, got = design_index(estimate), m is not None
            else:
                queue.append((t + d, m))
                beam = design_index(prior)
                estimate, sigma = tracker.step(late)
                got = late is not None
            w = ctx.legacy_beams[beam]
        else:
            beam = design_index(prior)
            w = ctx.legacy_beams[beam]
            amp = coop_observe(ctx.truth, w, x, g, ph, nz[30], imp)
            window = np.clip(3 * np.sqrt(tracker.cov[0, 0]) + .010, .025, .110) if tracker.initialized else .110
            m, _ = match(amp, ctx.legacy_templates[beam], imp.pilot_noise,
                         prior if tracker.initialized else 0., window)
            m = None if rr.lost[t] else m
            if d == 0:                    # original timing: tracker updated with this frame
                estimate, sigma = tracker.step(m)
                got = m is not None
            else:                         # naive: late report applied as if current
                queue.append((t + d, m))
                estimate, sigma = tracker.step(late)
                got = late is not None
        misses = 0 if got else misses + 1
        power = payload_power(ctx.truth, w, x, g, ph, imp)
        rows.append(_row(ctx, t, x, float(GRID_X[beam]), float(sigma), scanned, overhead, power))
    return rows


# ------------------------------------------------------------ passive runs
def run_passive(ctx: SceneContext, policy: str, path: np.ndarray, rr: RunRandom,
                rho: float, seed: int) -> list[dict]:
    """Non-cooperative receiver; the transmitter senses with echoes."""
    cfg = ctx.cond.echo
    if policy.endswith("_nogating"):
        cfg = replace(cfg, bandwidth_hz=0.0)
    rng = np.random.default_rng(seed)
    echo = EchoModel(ctx.truth, ctx.nominal, cfg, ctx.imp, rng)
    rho_grid = 10 ** (RHO_DB / 10)
    filt = GridFilter(FilterConfig(), n_aux=len(rho_grid))
    log_pi = ctx.log_pi0.copy()
    if policy.startswith("sweep21"):
        sweep = [D.focus_beam(ctx.nominal, xx) for xx in PASSIVE_LEGACY_X]
    else:
        sweep = ctx.scan_beams
    sweep_t = [[echo_template(h, w) for w in sweep] for h in ctx.hyps]
    same_waveform = policy.startswith("same_waveform")
    rows, prev = [], None

    def sense(w_list, templ_by_hyp, integration):
        nonlocal log_pi
        ys = [echo.observe(w, x, rho, integration) for w in w_list]
        ll_h = np.zeros((len(ctx.hyps), len(rho_grid), len(X_GRID)))
        for h in range(len(ctx.hyps)):
            for w, tp, y in zip(w_list, templ_by_hyp[h], ys):
                ll_h[h] += echo.loglik(tp, y, echo.interference_power(w, integration), rho_grid)
        flat = ll_h.reshape(len(ctx.hyps), -1)
        prior = filt.p.sum(axis=1).reshape(-1)
        log_pi = update_geometry(log_pi, flat, prior)
        ll = combine_hypotheses(flat, log_pi).reshape(len(rho_grid), len(X_GRID))
        filt.update(ll)

    for t, x in enumerate(path):
        if t:
            filt.predict()
        _, sd = filt.mean_std()
        scanned = (not same_waveform) or t == 0 or sd > RESCAN_STD_M
        overhead = TAU + (len(sweep) * TAU if scanned else 0.0)
        if scanned:
            sense(sweep, sweep_t, TAU)
        est, est_sd = filt.mean_std()
        w, _ = design_beam(ctx, filt.marginal_x(), log_pi, "phase", prev)
        prev = w
        g, ph = rr.gains[t], rr.phases[t]
        power = payload_power(ctx.truth, w, x, g, ph, ctx.imp)
        if same_waveform:
            payload_s = FRAME_S - overhead
            sense([w], [[echo_template(h, w)] for h in ctx.hyps], payload_s)
        rows.append(_row(ctx, t, x, est, est_sd, scanned, overhead, power))
    return rows
