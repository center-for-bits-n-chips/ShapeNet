#!/usr/bin/env python3
# membrane_segmented_electrodes.py

import torch
import numpy as np
from scipy.special import jn_zeros
import matplotlib.pyplot as plt

class PseudoArcLengthContinuation:
    def __init__(
        self,
        zeros_list,
        ds: float = 0.01,
        max_steps: int = 100,
        newton_tol: float = 1e-6,
        Nr: int = 200,
        Nθ: int = 400,
        electrode_masks: list = None,
        v_electrodes: list = None,
        dtype=torch.float64,
        device=torch.device('cpu'),
        save_every: int = 100
    ):
        # continuation parameters
        self.ds         = ds
        self.max_steps  = max_steps
        self.newton_tol = newton_tol
        self.save_every = save_every

        # build polar grid
        self.r_vals = torch.linspace(1e-6, 1.0, Nr, dtype=dtype, device=device)
        self.θ_vals = torch.linspace(0.0, 2*torch.pi, Nθ, dtype=dtype, device=device)
        self.dr     = self.r_vals[1] - self.r_vals[0]
        self.dθ     = self.θ_vals[1] - self.θ_vals[0]
        R, Θ = torch.meshgrid(self.r_vals, self.θ_vals, indexing='ij')
        self.R, self.Θ = R, Θ  # for evaluation

        # process segmented electrodes
        self.has_electrodes = False
        if electrode_masks is not None and v_electrodes is not None:
            if len(electrode_masks) != len(v_electrodes):
                raise ValueError("Length of electrode_masks must match length of v_electrodes")
            # stack masks into a tensor of shape (n_seg, Nr, Nθ)
            self.electrode_masks = torch.stack(
                [torch.tensor(mask, dtype=dtype, device=device) for mask in electrode_masks],
                dim=0
            )
            # voltages per segment
            self.v_electrodes = torch.tensor(v_electrodes, dtype=dtype, device=device)
            self.has_electrodes = True

        # build basis
        basis = []
        for n in zeros_list:
            basis.extend(self.make_pytorch_basis(n))
        self.basis = basis
        self.N     = len(basis)

        # precompute stiffness matrix
        self.K  = self._compute_stiffness_matrix()
        self.Kt = torch.tensor(self.K, dtype=dtype, device=device)

    @staticmethod
    def make_pytorch_basis(n):
        alpha0 = float(jn_zeros(0, n)[-1])
        alpha1 = float(jn_zeros(1, n)[-1])

        def phi0(r, θ=None):
            return torch.special.bessel_j0(alpha0 * r)

        def phi1_cos(r, θ):
            return torch.special.bessel_j1(alpha1 * r) * torch.cos(θ)

        def phi1_sin(r, θ):
            return torch.special.bessel_j1(alpha1 * r) * torch.sin(θ)

        return [
            {'kind': 'j0',      'alpha': alpha0, 'fn': phi0},
            {'kind': 'j1_cos',  'alpha': alpha1, 'fn': phi1_cos},
            {'kind': 'j1_sin',  'alpha': alpha1, 'fn': phi1_sin},
        ]

    def _compute_vk_tensor(self, a_t):
        """
        Differentiable von‐Kármán nonlinear vector N(a) (shape (N,)).
        """
        # 1) compute ∇w on the grid
        Dr_w = torch.zeros_like(self.R)
        Dθ_w = torch.zeros_like(self.R)
        for i, entry in enumerate(self.basis):
            a_i, alpha, kind = a_t[i], entry['alpha'], entry['kind']
            if kind == 'j0':
                Dr_m = -alpha * torch.special.bessel_j1(alpha * self.R)
                Dθ_m = torch.zeros_like(self.R)
            elif kind == 'j1_cos':
                J1 = torch.special.bessel_j1(alpha * self.R)
                J0 = torch.special.bessel_j0(alpha * self.R)
                Dr_m = (alpha * J0 - J1/self.R) * torch.cos(self.Θ)
                Dθ_m = -J1 * torch.sin(self.Θ)
            elif kind == 'j1_sin':
                J1 = torch.special.bessel_j1(alpha * self.R)
                J0 = torch.special.bessel_j0(alpha * self.R)
                Dr_m = (alpha * J0 - J1/self.R) * torch.sin(self.Θ)
                Dθ_m = J1 * torch.cos(self.Θ)
            else:
                raise ValueError(f"Unknown basis kind {kind}")
            Dr_w += a_i * Dr_m
            Dθ_w += a_i * Dθ_m

        # 2) nonlinear strain‐energy term
        G = Dr_w**2 + (1.0/self.R**2) * Dθ_w**2

        # 3) project back onto each mode
        N_list = []
        for entry in self.basis:
            alpha, kind = entry['alpha'], entry['kind']
            if kind == 'j0':
                Dr_m = -alpha * torch.special.bessel_j1(alpha * self.R)
                Dθ_m = torch.zeros_like(self.R)
            elif kind == 'j1_cos':
                J1 = torch.special.bessel_j1(alpha * self.R)
                J0 = torch.special.bessel_j0(alpha * self.R)
                Dr_m = (alpha * J0 - J1/self.R) * torch.cos(self.Θ)
                Dθ_m = -J1 * torch.sin(self.Θ)
            elif kind == 'j1_sin':
                J1 = torch.special.bessel_j1(alpha * self.R)
                J0 = torch.special.bessel_j0(alpha * self.R)
                Dr_m = (alpha * J0 - J1/self.R) * torch.sin(self.Θ)
                Dθ_m = J1 * torch.cos(self.Θ)
            else:
                raise ValueError(f"Unknown basis kind {kind}")

            dot       = Dr_w*Dr_m + (1.0/self.R**2)*Dθ_w*Dθ_m
            integrand = dot * G * self.R      # includes Jacobian r
            tmp       = torch.trapz(integrand, dx=self.dθ, dim=1)
            N_m       = 0.5 * torch.trapz(tmp,    dx=self.dr, dim=0)
            N_list.append(N_m)

        return torch.stack(N_list)  # shape (N,)

    def _compute_stiffness_matrix(self):
        """
        Diagonal stiffness:
        K[i,i] = ∫ [ (Dr)^2 + (1/r^2)*(Dθ)^2 ] * r dr dθ
        """
        K = np.zeros((self.N, self.N))
        for i, entry in enumerate(self.basis):
            alpha, kind = entry['alpha'], entry['kind']
            if kind == 'j0':
                Dr = -alpha * torch.special.bessel_j1(alpha * self.R)
                Dθ = torch.zeros_like(self.R)
            elif kind == 'j1_cos':
                J1 = torch.special.bessel_j1(alpha * self.R)
                J0 = torch.special.bessel_j0(alpha * self.R)
                Dr = (alpha * J0 - J1/self.R) * torch.cos(self.Θ)
                Dθ = -J1 * torch.sin(self.Θ)
            elif kind == 'j1_sin':
                J1 = torch.special.bessel_j1(alpha * self.R)
                J0 = torch.special.bessel_j0(alpha * self.R)
                Dr = (alpha * J0 - J1/self.R) * torch.sin(self.Θ)
                Dθ = J1 * torch.cos(self.Θ)
            else:
                raise ValueError(f"Unknown basis kind {kind}")
            integrand = (Dr**2 + (1.0/self.R**2)*Dθ**2) * self.R
            tmp = torch.trapz(integrand, dx=self.dθ, dim=1)
            Ki  = torch.trapz(tmp, dx=self.dr, dim=0)
            K[i,i] = Ki.item()
        return K

    def _compute_g_tensor(self, a_t, lam):
        """
        Electrostatic coupling with segmented electrodes:
        g_i = ∫ φ_i * r * lam_field / S^2  dr dθ
        lam_field = max(0, λ + Σ_i v_i·mask_i)
        """
        # stack φ: (Nr, Nθ, N)
        Phi = torch.stack([entry['fn'](self.R, self.Θ) for entry in self.basis], dim=2)
        # denominator S = 1 - Σ a_j φ_j
        S = 1.0 - torch.tensordot(Phi, a_t, dims=([2],[0]))
        if self.has_electrodes:
            lam_field = lam + torch.tensordot(self.electrode_masks, self.v_electrodes, dims=([0],[0]))
            lam_field = torch.clamp(lam_field, min=0.0)
            integrand = Phi * (self.R.unsqueeze(2) * lam_field.unsqueeze(2) / (S**2).unsqueeze(2))
        else:
            integrand = Phi * (self.R.unsqueeze(2) / (S**2).unsqueeze(2))

        tmp = torch.trapz(integrand, dx=self.dθ, dim=1)
        g_t = torch.trapz(tmp, dx=self.dr, dim=0)
        return g_t

    def residual(self, a, lam):
        a_t   = torch.tensor(a, dtype=self.Kt.dtype, device=self.Kt.device)
        N_t   = self._compute_vk_tensor(a_t)
        lam_t = torch.tensor(lam, dtype=self.Kt.dtype, device=self.Kt.device)
        g_t   = self._compute_g_tensor(a_t, lam_t)
        return self.K.dot(a) + N_t.cpu().numpy() - g_t.cpu().numpy()

    def jacobian(self, a, lam):
        x0 = torch.tensor(np.concatenate([a, [lam]]),
                          dtype=self.Kt.dtype, device=self.Kt.device,
                          requires_grad=True)

        def F_torch(x):
            a_t   = x[:-1]
            lam_t = x[-1]
            Ka    = self.Kt @ a_t
            Nvk   = self._compute_vk_tensor(a_t)
            gvec  = self._compute_g_tensor(a_t, lam_t)
            return Ka + Nvk - gvec

        J = torch.autograd.functional.jacobian(F_torch, x0)
        J_np = J.detach().cpu().numpy()
        return J_np[:, :-1], J_np[:, -1]

    def compute_tangent(self, a, lam):
        F_a, F_lam = self.jacobian(a, lam)
        A = np.hstack([F_a, F_lam.reshape(-1,1)])  # Nx(N+1)
        _,_,Vt = np.linalg.svd(A)
        t = Vt.conj().T[:,-1]
        if t[0] < 0: t = -t
        return t / np.linalg.norm(t)

    def continue_(self, a0, lam0):
        path = []
        a, lam = a0.copy(), lam0
        t = self.compute_tangent(a, lam)

        for i in range(self.max_steps):
            if i % self.save_every == 0:
                print(f"[iter {i}] λ = {lam:.6f}")
            x_prev = np.concatenate([a, [lam]])
            x_pred = x_prev + self.ds * t

            # Newton corrector with augmented system
            x = x_pred.copy()
            for _ in range(20):
                a_k, lam_k = x[:-1], x[-1]
                Fvec       = self.residual(a_k, lam_k)
                Φ          = t.dot(x - x_prev) - self.ds
                Faug       = np.concatenate([Fvec, [Φ]])

                F_a, F_l = self.jacobian(a_k, lam_k)
                Jaug = np.zeros((self.N+1, self.N+1))
                Jaug[:self.N, :self.N] = F_a
                Jaug[:self.N,   -1]    = F_l
                Jaug[-1, :self.N]      = t[:-1]
                Jaug[-1,   -1]         = t[-1]

                delta = np.linalg.solve(Jaug, -Faug)
                x += delta
                if np.linalg.norm(delta) < self.newton_tol:
                    break

            a, lam = x[:-1], x[-1]
            t      = self.compute_tangent(a, lam)
            if i % self.save_every == 0:
                path.append((a.copy(), lam))

        return path

    def reconstruct_shape(self, a):
        u = torch.zeros_like(self.R)
        for i, entry in enumerate(self.basis):
            u += a[i] * entry['fn'](self.R, self.Θ)
        return u

    def get_max_displacement(self, a):
        u = self.reconstruct_shape(a)
        return torch.max(torch.abs(u)).item()


def main():
    # Grid resolution
    Nr, Nθ = 200, 400

    # Build mesh for masks
    r_vals = np.linspace(1e-6, 1.0, Nr)
    θ_vals = np.linspace(0.0, 2*np.pi, Nθ, endpoint=False)
    R, Θ = np.meshgrid(r_vals, θ_vals, indexing='ij')

    # Define concentric ring electrodes
    ring_bounds = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]
    electrode_masks = []
    for r_min, r_max in ring_bounds:
        mask = ((R >= r_min) & (R < r_max)).astype(float)
        electrode_masks.append(mask)

    # Voltages for each ring (non-dimensional)
    v_electrodes = [ 0.05, -0.02, -0.05 ]

    # Choose Bessel‐zero modes
    zeros_list = [1, 2, 3]  # yields 3×3 = 9 basis functions

    # Instantiate solver
    cont = PseudoArcLengthContinuation(
        zeros_list     = zeros_list,
        ds             = 0.01,
        max_steps      = 200,
        newton_tol     = 1e-6,
        Nr             = Nr,
        Nθ             = Nθ,
        electrode_masks= electrode_masks,
        v_electrodes   = v_electrodes,
        save_every     = 10
    )

    # Initial (flat) solution
    a0, lam0 = np.zeros(cont.N), 0.0

    # Run continuation
    path = cont.continue_(a0, lam0)

    # Extract bifurcation data
    lam_vals    = [p[1] for p in path]
    max_disp    = [cont.get_max_displacement(p[0]) for p in path]

    # Plot bifurcation curve
    plt.figure(figsize=(8,5))
    plt.plot(lam_vals, max_disp, 'o-')
    plt.xlabel('Applied voltage λ')
    plt.ylabel('Max membrane displacement')
    plt.title('Bifurcation Diagram')
    plt.grid(True)
    plt.show()

    # Plot final membrane shape
    final_a     = path[-1][0]
    u           = cont.reconstruct_shape(final_a)
    X_cart      = cont.R.cpu().numpy() * np.cos(cont.Θ.cpu().numpy())
    Y_cart      = cont.R.cpu().numpy() * np.sin(cont.Θ.cpu().numpy())

    plt.figure(figsize=(6,6))
    plt.contourf(X_cart, Y_cart, u.cpu().numpy(), levels=30)
    plt.colorbar(label='Displacement')
    plt.xlabel('x'); plt.ylabel('y')
    plt.axis('equal')
    plt.title(f'Final Shape at λ = {path[-1][1]:.3f}')
    plt.show()


if __name__ == "__main__":
    main()
