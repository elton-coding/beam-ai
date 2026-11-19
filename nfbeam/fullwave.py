"""Two-dimensional full-wave channel by a boundary integral method.

The angular-spectrum operator in ``nfbeam.channel`` treats the obstacle as a
stack of thin absorbing masks: no reflection, approximate edge diffraction and
no polarization.  This module solves the 2-D Helmholtz equation exactly for a
real obstacle boundary, independently of that approximation:

* incident field: each array element is a uniform aperture segment on z = 0,
  radiated by the 2-D Rayleigh–Sommerfeld integral (identical to exact
  angular-spectrum propagation in free space);
* obstacle: a closed polygon with one of three boundary conditions,
    ``"tm"``     perfect conductor, E out of plane  (Dirichlet, u = 0),
    ``"te"``     perfect conductor, H out of plane  (Neumann, du/dn = 0),
    ``"absorber"`` impedance surface (Robin, du/dn = -ik beta u); beta = 1 is
                 matched (no specular reflection), beta < 1 a partly reflecting
                 lossy body with normal-incidence reflection (1 - beta)/(1 + beta);
* solver: direct (Green's representation) boundary integral equation,
  pulse basis, midpoint collocation, singular self terms in closed form,
  refined quadrature for near pairs, CHIEF interior points against the
  interior resonances of an electrically large body (least squares).

Time convention exp(-i w t), outgoing Hankel functions H^(1), as in the
angular-spectrum engine (transfer function exp(+i kz z)).
"""

from __future__ import annotations

import numpy as np
from scipy.special import hankel1

from direct_network.core.physics import Scene
from nfbeam.channel import Channel

EULER_GAMMA = 0.5772156649015329
BCS = ("tm", "te", "absorber")


def _gauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    s, w = np.polynomial.legendre.leggauss(n)
    return (s + 1) / 2, w / 2          # on [0, 1]


class Boundary:
    """Closed polygon split into straight segments of length <= seg_len.

    Vertices are given counter-clockwise in the (x, z) plane, so the outward
    normal of a segment with unit tangent (tx, tz) is (tz, -tx).
    """

    def __init__(self, vertices: np.ndarray, seg_len: float):
        v = np.asarray(vertices, dtype=float)
        a, b = [], []
        for p, q in zip(v, np.roll(v, -1, axis=0)):
            n = max(1, int(np.ceil(np.hypot(*(q - p)) / seg_len)))
            t = np.arange(n + 1)[:, None] / n
            pts = p + t * (q - p)
            a.append(pts[:-1])
            b.append(pts[1:])
        self.a = np.concatenate(a)
        self.b = np.concatenate(b)
        d = self.b - self.a
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.tangent = d / self.length[:, None]
        self.normal = np.stack([self.tangent[:, 1], -self.tangent[:, 0]], axis=1)
        self.mid = (self.a + self.b) / 2
        self.n = len(self.length)
        area = 0.5 * np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])
        if area <= 0:
            raise ValueError("vertices must be counter-clockwise")
        self.vertices = v

    def interior_points(self, count: int, margin: float, rng: np.random.Generator) -> np.ndarray:
        from matplotlib.path import Path
        lo, hi = self.vertices.min(0), self.vertices.max(0)
        path = Path(self.vertices)
        out = []
        while len(out) < count:
            p = rng.uniform(lo, hi, size=(4 * count, 2))
            inside = path.contains_points(p)
            dist = np.min(np.hypot(p[:, None, 0] - self.mid[None, :, 0],
                                   p[:, None, 1] - self.mid[None, :, 1]), axis=1)
            out.extend(p[inside & (dist > margin)])
        return np.asarray(out[:count])


def rectangle(x_c: float, z_c: float, width: float, depth: float) -> np.ndarray:
    x0, x1 = x_c - width / 2, x_c + width / 2
    z0, z1 = z_c - depth / 2, z_c + depth / 2
    return np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]])


def circle(radius: float, n_vertices: int, center=(0.0, 0.0)) -> np.ndarray:
    t = 2 * np.pi * np.arange(n_vertices) / n_vertices
    return np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], axis=1)


# --------------------------------------------------------------------- layers
def layer_matrices(obs: np.ndarray, bnd: Boundary, k: float, nq: int = 2,
                   chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Single- and double-layer matrices from boundary segments to points.

    S[i, j] = int_j G(r_i, r') dl',   D[i, j] = int_j dG(r_i, r')/dn' dl',
    with G = (i/4) H0^(1)(k |r - r'|).
    """
    s, w = _gauss(nq)
    obs = np.asarray(obs, dtype=float)
    S = np.empty((len(obs), bnd.n), dtype=complex)
    D = np.empty_like(S)
    d = bnd.b - bnd.a
    for lo in range(0, len(obs), chunk):
        o = obs[lo:lo + chunk]
        s_acc = np.zeros((len(o), bnd.n), dtype=complex)
        d_acc = np.zeros_like(s_acc)
        for sq, wq in zip(s, w):
            p = bnd.a + sq * d
            rx = o[:, None, 0] - p[None, :, 0]
            rz = o[:, None, 1] - p[None, :, 1]
            rho = np.hypot(rx, rz)
            kr = k * rho
            s_acc += wq * hankel1(0, kr)
            proj = (rx * bnd.normal[None, :, 0] + rz * bnd.normal[None, :, 1]) / rho
            d_acc += wq * k * hankel1(1, kr) * proj
        S[lo:lo + chunk] = 0.25j * s_acc * bnd.length[None, :]
        D[lo:lo + chunk] = 0.25j * d_acc * bnd.length[None, :]
    return S, D


def boundary_matrices(bnd: Boundary, k: float, near_factor: float = 4.0,
                      nq_far: int = 2, nq_near: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Collocation matrices on the boundary itself (principal value for D)."""
    S, D = layer_matrices(bnd.mid, bnd, k, nq_far)
    # Near pairs: refined quadrature.
    dist = np.hypot(bnd.mid[:, None, 0] - bnd.mid[None, :, 0], bnd.mid[:, None, 1] - bnd.mid[None, :, 1])
    near_i, near_j = np.nonzero((dist < near_factor * bnd.length.max()) & ~np.eye(bnd.n, dtype=bool))
    if len(near_i):
        s, w = _gauss(nq_near)
        o = bnd.mid[near_i]
        a, b = bnd.a[near_j], bnd.b[near_j]
        nrm = bnd.normal[near_j]
        s_acc = np.zeros(len(near_i), dtype=complex)
        d_acc = np.zeros_like(s_acc)
        for sq, wq in zip(s, w):
            p = a + sq * (b - a)
            r = o - p
            rho = np.hypot(r[:, 0], r[:, 1])
            s_acc += wq * hankel1(0, k * rho)
            d_acc += wq * k * hankel1(1, k * rho) * np.sum(r * nrm, axis=1) / rho
        S[near_i, near_j] = 0.25j * s_acc * bnd.length[near_j]
        D[near_i, near_j] = 0.25j * d_acc * bnd.length[near_j]
    # Self terms: log singularity integrated in closed form; D vanishes on a flat segment.
    L = bnd.length
    S[np.arange(bnd.n), np.arange(bnd.n)] = 0.25j * L * (1 + 2j / np.pi * (np.log(k * L / 4) + EULER_GAMMA - 1))
    D[np.arange(bnd.n), np.arange(bnd.n)] = 0.0
    return S, D


# ---------------------------------------------------------------------- solver
class Scatterer:
    """Boundary integral solution for one obstacle and boundary condition."""

    def __init__(self, vertices: np.ndarray, k: float, bc: str = "tm",
                 points_per_wavelength: float = 10.0, chief_fraction: float = 0.1,
                 beta: float = 1.0, seed: int = 0):
        if bc not in BCS:
            raise ValueError(f"bc must be one of {BCS}")
        self.k, self.bc, self.beta = k, bc, beta
        lam = 2 * np.pi / k
        self.bnd = Boundary(vertices, lam / points_per_wavelength)
        S, D = boundary_matrices(self.bnd, k)
        n = self.bnd.n
        n_chief = max(20, int(chief_fraction * n))
        self.chief = self.bnd.interior_points(n_chief, margin=lam / 2, rng=np.random.default_rng(seed))
        Sc, Dc = layer_matrices(self.chief, self.bnd, k, nq=4)
        I = np.eye(n)
        # Total field u = u_inc + D u - S dn(u) outside; extinction inside.
        if bc == "tm":       # unknown q = dn(u); u = 0 on the boundary
            A_b, A_c = S, Sc
        elif bc == "te":     # unknown u; dn(u) = 0
            A_b, A_c = 0.5 * I - D, -Dc
        else:                # unknown u; dn(u) = -i k beta u
            A_b, A_c = 0.5 * I - D - 1j * k * beta * S, -(Dc + 1j * k * beta * Sc)
        self._A = np.vstack([A_b, A_c])
        # QR once; solves for any number of right-hand sides.
        self._Q, self._R = np.linalg.qr(self._A)

    def rhs_points(self) -> np.ndarray:
        """Points where the incident field is needed: collocation + CHIEF."""
        return np.vstack([self.bnd.mid, self.chief])

    def solve(self, u_inc: np.ndarray) -> np.ndarray:
        """Boundary density for incident fields sampled at ``rhs_points()``.

        ``u_inc`` has shape (n_points,) or (n_points, n_rhs).
        """
        from scipy.linalg import solve_triangular
        return solve_triangular(self._R, self._Q.conj().T @ u_inc)

    def scattered(self, obs: np.ndarray, density: np.ndarray, nq: int = 3) -> np.ndarray:
        """Scattered field at ``obs`` (n_obs,) or (n_obs, n_rhs)."""
        S, D = layer_matrices(obs, self.bnd, self.k, nq=nq)
        if self.bc == "tm":
            return -S @ density
        if self.bc == "te":
            return D @ density
        return (D + 1j * self.k * self.beta * S) @ density


# ------------------------------------------------------------ incident fields
def aperture_field(obs: np.ndarray, centers: np.ndarray, pitch: float, k: float,
                   nq: int = 4, chunk: int = 2048) -> np.ndarray:
    """Free-space field at ``obs`` (z > 0) of unit-amplitude aperture elements.

    2-D Rayleigh–Sommerfeld (first kind): u(r) = (i k / 2) int u0(x') H1(k rho) z / rho dx'.
    Returns shape (n_elements, n_obs).
    """
    s, w = _gauss(nq)
    obs = np.asarray(obs, dtype=float)
    out = np.zeros((len(centers), len(obs)), dtype=complex)
    for lo in range(0, len(obs), chunk):
        o = obs[lo:lo + chunk]
        acc = np.zeros((len(centers), len(o)), dtype=complex)
        for sq, wq in zip(s, w):
            xp = centers - pitch / 2 + sq * pitch
            rx = o[None, :, 0] - xp[:, None]
            z = o[None, :, 1]
            rho = np.hypot(rx, z)
            acc += wq * hankel1(1, k * rho) * z / rho
        out[:, lo:lo + chunk] = 0.5j * k * pitch * acc
    return out


def line_source_field(obs: np.ndarray, src: np.ndarray, k: float) -> np.ndarray:
    obs = np.asarray(obs, dtype=float)
    rho = np.hypot(obs[:, 0] - src[0], obs[:, 1] - src[1])
    return 0.25j * hankel1(0, k * rho)


# ---------------------------------------------------------------------- channel
class FullWaveChannel(Channel):
    """``Channel`` whose element-to-receiver operator is the full-wave solution.

    All beam and scoring methods are inherited unchanged, so any design or
    experiment can swap the angular-spectrum operator for this one.  ``H_as``
    keeps the angular-spectrum operator of the same geometry for comparison.
    ``H_echo`` is the monostatic scattered field at the element centres
    (n_tx x n_tx), i.e. what the array would receive back from the obstacle.
    """

    def __init__(self, scene: Scene, bc: str = "tm", points_per_wavelength: float = 10.0,
                 keep_as: bool = False, echo: bool = False, beta: float = 1.0):
        super().__init__(scene)
        self.bc, self.beta = bc, beta
        if keep_as:
            self.H_as = self.H.copy()
        za_c = scene.obstacle_z_m
        verts = rectangle(scene.obstacle_x_m, za_c, scene.obstacle_width_m, scene.obstacle_depth_m)
        self.scatterer = Scatterer(verts, self.k, bc, points_per_wavelength, beta=beta)
        rx = np.stack([self.x, np.full_like(self.x, scene.receiver_z_m)], axis=1)
        u_bnd = aperture_field(self.scatterer.rhs_points(), self.centers, self.pitch, self.k)
        density = self.scatterer.solve(u_bnd.T)                       # (n_unknown, n_tx)
        self.H = np.ascontiguousarray(aperture_field(rx, self.centers, self.pitch, self.k)
                                      + self.scatterer.scattered(rx, density).T)
        if echo:
            # Scattered field back on the aperture plane, sampled at the element
            # centres: H_echo[i, j] = field at element j when element i transmits.
            # With receive weights equal to the transmit weights the coherent
            # echo is  y = pitch * w^T H_echo w  (same units as one-way power).
            arr = np.stack([self.centers, np.full_like(self.centers, 1e-4)], axis=1)
            self.H_echo = self.scatterer.scattered(arr, density).T     # (n_tx, n_rx)
        self.n_unknowns = self.scatterer.bnd.n
