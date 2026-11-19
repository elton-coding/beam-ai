"""Unit tests for nfbeam.  Run with ``python -m pytest tests`` or ``python -m tests``."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from direct_network.core.physics import AngularSpectrum, Scene
from nfbeam import design as D
from nfbeam.channel import Channel
from nfbeam.link import make_losses
from nfbeam.sensing import Impairments, coop_loglik, coop_observe, coop_template, hann_range_response
from nfbeam.tracking import X_GRID, DelayedFilter, FilterConfig, GridFilter

SCENE = replace(Scene(), n_grid=8192, obstacle_step_m=0.0015)
_CH = None


def channel() -> Channel:
    global _CH
    if _CH is None:
        _CH = Channel(SCENE)
    return _CH


def test_channel_matches_reference_engine():
    ch, eng = channel(), AngularSpectrum(SCENE)
    for b, f, th, x in [(0, 0.36, 0, 0.0), (4, 0.26, 5, 0.01), (0, 0.37, 14, 0.095)]:
        old = eng.received_power(x, b, f, th)
        new = ch.power(D.airy_beam(ch, b, f, th), x)
        assert abs(new / old - 1) < 0.05, (b, x, old, new)


def test_unit_transmit_power_and_passivity():
    ch = channel()
    w = D.focus_beam(ch, 0.0)
    assert abs(ch.tx_power(w) - 1) < 1e-12
    total = float(np.sum(np.abs(ch.field(w)) ** 2) * ch.dx)
    assert total <= 1.0 + 1e-9


def test_quadratic_form_equals_power():
    ch = channel()
    w = D.airy_beam(ch, 4, 0.26, 5)
    assert abs(D.quad(ch.gram_at(0.01), w) - ch.power(w, 0.01)) < 1e-12


def test_phase_only_improves_and_is_bounded():
    ch = channel()
    x = 0.012
    A = ch.gram_at(x)
    seed = D.focus_beam(ch, x)
    w = D.phase_only(ch, A, seed)
    ceil, _ = D.ceiling(ch, A)
    assert D.quad(A, w) >= D.quad(A, seed) - 1e-12
    assert D.quad(A, w) <= ceil * (1 + 1e-9)
    assert np.allclose(np.abs(w), np.abs(w[0]), rtol=1e-9)        # constant amplitude


def test_robust_gram_is_expectation():
    ch = channel()
    xs, ps = np.array([0.0, 0.01]), np.array([0.3, 0.7])
    A = D.robust_gram([ch, ch], np.array([0.5, 0.5]), xs, ps)
    ref = 0.3 * ch.gram_at(0.0) + 0.7 * ch.gram_at(0.01)
    assert np.allclose(A, ref, atol=1e-12)


def test_filter_conserves_mass_and_tracks():
    f = GridFilter(FilterConfig())
    rng = np.random.default_rng(0)
    x, errs = -0.05, []
    for k in range(60):
        if k:
            f.predict()
            x += 0.1 * 0.02
        z = x + rng.normal(0, 0.002)
        f.update(-0.5 * ((X_GRID - z) / 0.002) ** 2)
        assert abs(f.p.sum() - 1) < 1e-9
        errs.append(abs(f.mean_std()[0] - x))
    assert np.mean(errs[10:]) < 0.003


def test_delayed_update_equals_in_order_update():
    lls = [-0.5 * ((X_GRID - (0.01 + 0.002 * k)) / 0.003) ** 2 for k in range(6)]
    a = DelayedFilter(GridFilter(FilterConfig()))
    b = DelayedFilter(GridFilter(FilterConfig()))
    for t in range(6):
        a.advance()
        a.add(t, lls[t])
        b.advance()
        if t >= 2:
            b.add(t - 2, lls[t - 2])           # two frames late
    for s in (4, 5):
        b.add(s, lls[s])
    assert np.allclose(a.filt.p, b.filt.p, atol=1e-10)


def test_cooperative_likelihood_peaks_at_truth():
    ch = channel()
    imp = Impairments(pilot_noise=1e-8, gain_std_db=0.0, multipath=0.0)
    w = D.focus_beam(ch, 0.03)
    y = coop_observe(ch, w, 0.031, 1.0, 0.0, np.zeros(4), imp)
    ll = coop_loglik([coop_template(ch, w)], y, Impairments(), 3e-4)[0]
    assert abs(X_GRID[np.argmax(ll)] - 0.031) <= 0.0015


def test_range_response():
    assert abs(hann_range_response(0.0) - 1) < 1e-12
    assert hann_range_response(4.0) < 1e-2


def test_loss_processes():
    rng = np.random.default_rng(1)
    assert not make_losses("none", 1000, rng).any()
    assert abs(make_losses("iid20", 20000, rng).mean() - 0.2) < 0.02
    burst = make_losses("burst", 20000, rng)
    assert 0.05 < burst.mean() < 0.2


def test_measurement_schema_roundtrip(tmp_path=None):
    import tempfile
    from pathlib import Path
    from nfbeam.calibration import load_measurements, save_measurements
    d = Path(tmp_path or tempfile.mkdtemp())
    meas = pd.DataFrame({"rx_x_m": [0.0], "beam_id": [0], "subaperture": [0],
                         "amplitude": [0.1], "noise_power": [1e-5]})
    save_measurements(d, meas, {0: np.zeros(160)})
    m2, beams = load_measurements(d)
    assert len(m2) == 1 and beams[0].shape == (160,)


def test_power_operator_matches_robust_gram():
    ch = channel()
    hyps = [ch, Channel(replace(SCENE, obstacle_x_m=0.003))]
    pi = np.array([0.7, 0.3])
    px = np.exp(-0.5 * ((X_GRID - 0.01) / 0.003) ** 2)
    keep = px > 1e-4 * px.max()
    xs, ps = X_GRID[keep], px[keep] / px[keep].sum()
    A = D.robust_gram(hyps, pi, xs, ps)
    op = D.PowerOperator(hyps, pi, xs, ps)
    w = D.airy_beam(ch, 4, 0.26, 5)
    assert abs(op.quad(w) / D.quad(A, w) - 1) < 1e-10
    assert np.allclose(op @ w, A @ w, rtol=1e-8, atol=1e-14)
    coarse = D.PowerOperator(hyps, pi, xs, ps, max_columns=200)
    assert coarse.H.shape[1] <= 200 + 2
    assert abs(coarse.quad(w) / op.quad(w) - 1) < 0.05        # decimated quadrature stays close


def test_warm_start_never_worse_than_its_seeds():
    ch = channel()
    x = 0.005
    A = ch.gram_at(x)
    op = D.PowerOperator([ch], np.array([1.0]), np.array([x]), np.array([1.0]))
    seeds = [D.focus_beam(ch, x), D.airy_beam(ch, 4, 0.26, 5)]
    w = D.warm_phase_only(ch, op, seeds, iters=3)
    assert D.quad(A, w) >= max(D.quad(A, s) for s in seeds) - 1e-12


def test_wide_airy_search_not_worse_than_local_search():
    sc = replace(SCENE, tx_width_m=0.1)
    ch = Channel(sc)
    for x in (0.0, 0.03):
        A = ch.gram_at(x)
        assert D.quad(A, D.airy_search_wide(ch, A, x)) >= D.quad(A, D.airy_search(ch, A, x, rescale=True)) - 1e-12
