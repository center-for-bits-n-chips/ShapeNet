# membrane_shape_reconstruction.py
"""Real-time membrane shape reconstruction with physics-aware Tikhonov regularisation

Author: John Z. Zhang
Date: 2025-06-04

This script demonstrates how to
1. build a Bessel-function modal basis that satisfies a clamped edge (w=0 at r=1),
2. assemble the probe-to-mode measurement matrix Φ for an arbitrary set of probe
   coordinates (x, y),
3. apply a *physics-aware* ridge penalty that weights each mode by its
   strain-energy / natural-frequency cost (ω_j ∝ k_{n,m}² for a tensioned membrane),
4. pre-compute all heavy linear-algebra factors *offline*,
5. perform a lightweight (≈15x15) solve *online* each frame to recover the modal
   coefficient vector `a` from the latest probe readings `z`.

The code works for any number of probes N (≥1) and any number of modes M.  When
M>N the regularisation eliminates the null-space and returns the minimum-energy
shape consistent with the measurements.

Dependencies: numpy, scipy (for Bessel functions)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.special import jn, jn_zeros

from read_dat import read_labview_binary

# -----------------------------------------------------------------------------
# MODE DEFINITIONS & BASIS GENERATION
# -----------------------------------------------------------------------------

@dataclass
class Mode:
    """Single (n, m) Bessel mode with either cos(nθ) or sin(nθ) angular part."""

    n: int               # angular order
    k: float             # k_{n,m} satisfying J_n(k) = 0
    kind: str            # 'cos' or 'sin'
    omega: float         # energy weighting (k^2 for tensioned membrane)

    def phi(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Evaluate this mode at polar coords (r, θ)."""
        base = jn(self.n, self.k * r)
        if self.kind == "cos":
            return base * np.cos(self.n * theta)
        else:
            return base * np.sin(self.n * theta)


def build_modes(max_n: int = 3, max_m: int = 3) -> List[Mode]:
    """Generate Bessel modes up to n ≤ max_n and the first max_m radial roots.

    Returns a list of Mode objects.
    """
    modes: List[Mode] = []
    for n in range(max_n + 1):
        roots = jn_zeros(n, max_m)
        for k in roots:
            # cos term
            modes.append(Mode(n, k, "cos", omega=k**2))
            # add sin term for non-axisymmetric n≥1
            if n > 0:
                modes.append(Mode(n, k, "sin", omega=k**2))
    return modes

# -----------------------------------------------------------------------------
# MEASUREMENT MATRIX
# -----------------------------------------------------------------------------

def cartesian_to_polar(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (x, y) array → (r, θ)."""
    x, y = xy[:, 0], xy[:, 1]
    r = np.hypot(x, y) / 250.0
    theta = np.arctan2(y, x)
    return r, theta

def build_measurement_matrix(probes_xy: np.ndarray, modes: List[Mode]) -> np.ndarray:
    """Assemble Φ where Φ[i,j] = φ_j(r_i, θ_i)."""
    r, theta = cartesian_to_polar(probes_xy)
    N = probes_xy.shape[0]
    M = len(modes)
    Phi = np.empty((N, M), dtype=float)
    for j, mode in enumerate(modes):
        Phi[:, j] = mode.phi(r, theta)
    return Phi

# -----------------------------------------------------------------------------
# OFFLINE PRE-COMPUTATION (PHYSICS-AWARE RIDGE FACTORS)
# -----------------------------------------------------------------------------

@dataclass
class OfflineFactors:
    """Container holding pre-computed matrices for fast online reconstruction."""
    W_inv: np.ndarray         # diag(1 / sqrt(ω_j)) (M, M)
    Phi_tilde: np.ndarray     # Φ·W_inv              (N, M)
    G: np.ndarray             # Φ~ Φ~ᵀ               (N, N)
    lam: float                # regularisation gain
    modes: List[Mode]         # basis (for field reconstruction)


def precompute_factors(Phi: np.ndarray, modes: List[Mode], lam: float = 1e-3) -> OfflineFactors:
    """Compute and store matrices needed for the dual-form online solve."""
    omega = np.array([m.omega for m in modes])
    W_inv = np.diag(1.0 / np.sqrt(omega))            # (M, M)
    Phi_tilde = Phi @ W_inv                          # (N, M)
    G = Phi_tilde @ Phi_tilde.T                      # (N, N)
    return OfflineFactors(W_inv=W_inv, Phi_tilde=Phi_tilde, G=G, lam=lam, modes=modes)

# -----------------------------------------------------------------------------
# REAL-TIME UPDATE
# -----------------------------------------------------------------------------

def recover_modes(z: np.ndarray, pre: OfflineFactors) -> np.ndarray:
    """Compute modal coefficients a from probe displacements z (length N)."""
    # Solve small N×N system: (G + λ² I) y = z
    N = pre.G.shape[0]
    rhs = np.linalg.solve(pre.G + pre.lam ** 2 * np.eye(N), z)
    # a = W_inv · Φ~ᵀ · y
    a = pre.W_inv @ (pre.Phi_tilde.T @ rhs)
    return a  # shape (M,)

# -----------------------------------------------------------------------------
# FIELD RECONSTRUCTION ON A GRID (OPTIONAL)
# -----------------------------------------------------------------------------

def reconstruct_field(a: np.ndarray, pre: OfflineFactors, grid_r: np.ndarray, grid_theta: np.ndarray) -> np.ndarray:
    """Return w(r, θ) on the provided polar grid."""
    w = np.zeros_like(grid_r)
    for coeff, mode in zip(a, pre.modes):
        w += coeff * mode.phi(grid_r, grid_theta)
    return w

# -----------------------------------------------------------------------------
# DEMO / SELF-TEST (run "python membrane_shape_reconstruction.py" to see it work)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ---------------------------- User-adjustable parameters ------------------
    N_PROBES = 14          # number of displacement probes
    MAX_N = 3              # highest angular order n
    MAX_M = 5              # number of radial roots per n (m = 1…MAX_M)
    LAMBDA = 1e-2          # Tikhonov weight
    num_markers = 14
    num_voltages = 11
    gap = 35.0
    depth = 24.0

    # # ---------------------------- Synthetic probe layout ----------------------
    # # Center point
    # probes_xy = [(0.0, 0.0)]

    # Read first file
    filename1 = "Lincoln Labs data/2025-06-06 DIC measurement final case B.dat"
    voltages1, positions1, time1 = read_labview_binary(filename1, num_markers=num_markers, num_voltages=num_voltages, decimate=1)
    
    end_index_1 = 38550

    probes_xy = np.array([(marker[0], marker[1]) for marker in positions1[end_index_1, :, :]])
    z_meas_non_dim = positions1[end_index_1, :, 2] / gap
    
    # Parameters for rings
    n_rings = 3  # Number of rings
    points_per_ring = 4  # Points per ring
    ring_spacing = 0.3  # Spacing between rings
    
    # Add points in concentric rings
    # for ring in range(n_rings):
    #     r = (ring + 1) * ring_spacing  # Radius of current ring
    #     theta_offset = ring * np.pi/4  # 45 degree offset per ring
    #     for point in range(points_per_ring):
    #         theta = 2 * np.pi * point / points_per_ring + theta_offset
    #         x = r * np.cos(theta)
    #         y = r * np.sin(theta)
    #         probes_xy.append((x, y))
            
    # # Convert to numpy array
    # probes_xy = np.array(probes_xy)

    # ---------------------------- Build basis & measurement matrix -----------
    modes = build_modes(MAX_N, MAX_M)              # M modes
    Phi = build_measurement_matrix(probes_xy, modes)

    # ---------------------------- OFFLINE stage ------------------------------
    pre = precompute_factors(Phi, modes, lam=LAMBDA)

    # ---------------------------- Project paraboloid onto modes -------------
    # Generate paraboloid shape on a fine grid for accurate projection
    N_proj = 100
    r_proj = np.linspace(0, 1, N_proj)
    th_proj = np.linspace(0, 2*np.pi, N_proj)
    R_proj, TH_proj = np.meshgrid(r_proj, th_proj, indexing='ij')
    
    # w(r) = -r²/2 (normalized paraboloid)
    # Plot the paraboloid first
    W_parab = -(depth/gap) * (R_proj**2 - 1)
    
    # Project onto modes by numerical integration
    a_true = np.zeros(len(modes))
    for i, mode in enumerate(modes):
        # To ensure the reconstruction matches W_parab, the coefficients a_true[i]
        # must be calculated as the projection of W_parab onto each basis mode phi_i.
        # For an orthogonal basis {phi_i}, the coefficient a_i for a function f is:
        # a_i = <f, phi_i> / <phi_i, phi_i>
        # The inner product <g, h> in polar coordinates (r, theta) is typically
        # integral(integral(g(r,th) * h(r,th) * r dr dth)).

        # Here, f is W_parab, and phi_i is mode.phi(R_proj, TH_proj).
        # The 'r' weighting factor for the integral measure is R_proj.

        # Numerator: <W_parab, phi_i>
        # integrand_numerator = W_parab * phi_i * r
        numerator_integrand = W_parab * mode.phi(R_proj, TH_proj) * R_proj
        
        # Numerically integrate: first over r (axis=0), then over theta.
        # Note: R_proj, TH_proj are from meshgrid(r_proj, th_proj, indexing='ij'),
        # so R_proj varies along axis 0, TH_proj varies along axis 1.
        integral_over_r_num = np.trapz(numerator_integrand, r_proj, axis=0)
        numerator = np.trapz(integral_over_r_num, th_proj)

        # Denominator: <phi_i, phi_i> (squared L2 norm of the mode)
        # integrand_denominator = phi_i^2 * r
        denominator_integrand = (mode.phi(R_proj, TH_proj))**2 * R_proj
        
        # Numerically integrate for the denominator
        integral_over_r_den = np.trapz(denominator_integrand, r_proj, axis=0)
        denominator = np.trapz(integral_over_r_den, th_proj)
        
        # Calculate the coefficient a_true[i]
        if denominator == 0:
            # This case might occur if a mode is identically zero or the integration
            # grid is inadequate, though unlikely for well-defined basis functions.
            a_true[i] = 0.0
        else:
            a_true[i] = numerator / denominator
    
    # Generate synthetic probe data by sampling paraboloid at probe locations
    # z_true = 0.5 * (np.sum(probes_xy**2, axis=1) - 1)  # w(r) = -r²/2 evaluated at probe points
    # noise = 2e-2 * np.random.standard_normal(len(z_true))
    # z_meas = z_true + noise
    # print("Measured probe values:", z_meas)

    # ---------------------------- ONLINE reconstruction ----------------------
    a_est = recover_modes(z_meas_non_dim, pre)

    # Plot the mode amplitudes
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(a_true)), a_true)
    plt.bar(range(len(a_est)), a_est)
    plt.xlabel('Mode index')
    plt.ylabel('Mode amplitude')
    plt.title('True mode amplitudes')
    plt.grid(True)
    plt.show()

    # plot field on a polar grid for visual sanity check (optional)
    N_GRID = 100
    r_lin = np.linspace(0, 1, N_GRID)
    th_lin = np.linspace(0, 2 * np.pi, N_GRID)
    R, TH = np.meshgrid(r_lin, th_lin, indexing="ij")
    W_true = reconstruct_field(a_true, pre, R, TH)
    W_est = reconstruct_field(a_est, pre, R, TH)

    fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(10, 4))
    error = W_true - W_est
    cs0 = axs[0].contourf(TH, R, error, levels=30)
    axs[0].set_title("Error (Ground truth - Reconstruction)")
    fig.colorbar(cs0, ax=axs[0])

    cs1 = axs[1].contourf(TH, R, W_est, levels=30)
    axs[1].set_title("Reconstruction")
    fig.colorbar(cs1, ax=axs[1])

    plt.tight_layout()
    plt.show()


    # Plot paraboloid and measurements in 3D
    fig = plt.figure(figsize=(7, 5))
    
    # Plot paraboloid and measurements
    ax = fig.add_subplot(projection='3d')
    X = 250 * R_proj * np.cos(TH_proj)
    Y = 250 * R_proj * np.sin(TH_proj)
    surf = ax.plot_surface(X, Y, W_est, cmap='viridis', alpha=0.8)
    ax.scatter(probes_xy[:,0], probes_xy[:,1], z_meas_non_dim, c='red')
    ax.set_title('Target Paraboloid with Measurements')
    fig.colorbar(surf, ax=ax)

    plt.show()
