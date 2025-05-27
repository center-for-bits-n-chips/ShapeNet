# arc_length_continuation.py
import torch
import numpy as np
from scipy.special import jn_zeros
import matplotlib.pyplot as plt

def make_pytorch_basis(n):
    """
    Returns three PyTorch‐based basis entries for the n-th zero:
      – axisymmetric:  ('j0',      alpha0, phi0)
      – first cosine:  ('j1_cos',  alpha1, phi1_cos)
      – first sine:    ('j1_sin',  alpha1, phi1_sin)
    where alpha_m = nth zero of J_m.
    """
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

class PseudoArcLengthContinuation:
    def __init__(
        self,
        zeros_list,
        ds: float = 0.01,
        max_steps: int = 200,
        newton_tol: float = 1e-6,
        Nr: int = 200,
        Nθ: int = 400,
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
        self.R, self.Θ = R, Θ  # tensors for evaluation

        # build basis
        basis = []
        for n in zeros_list:
            basis.extend(make_pytorch_basis(n))
        self.basis = basis
        self.N     = len(basis)

        # precompute stiffness matrix (numpy) with explicit derivatives
        self.K  = self._compute_stiffness_matrix()
        self.Kt = torch.tensor(self.K, dtype=dtype, device=device)

    def _compute_stiffness_matrix(self):
        """
        Diagonal stiffness:
          K[i,i] = ∫ [ (Dr)^2 + (1/r^2)*(Dθ)^2 ] * r dr dθ
        with analytic Dr, Dθ per mode kind.
        """
        K = np.zeros((self.N, self.N))
        for i, entry in enumerate(self.basis):
            kind  = entry['kind']
            alpha = entry['alpha']

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
                Dθ =  J1 * torch.cos(self.Θ)
            else:
                raise ValueError(f"Unknown basis kind {kind}")

            integrand = (Dr**2 + (1.0/self.R**2)*Dθ**2) * self.R
            tmp = torch.trapz(integrand, dx=self.dθ, dim=1)  # integrate θ
            Ki  = torch.trapz(tmp,       dx=self.dr, dim=0)  # integrate r
            K[i,i] = Ki.item()
        return K

    def _compute_g_tensor(self, a_torch):
        """
        Electrostatic coupling vector:
          g_i = ∫ φ_i * r / (1 + Σ a_j φ_j)^2  dr dθ
        returns torch.Tensor of shape (N,).
        """
        # stack φ values: shape (Nr, Nθ, N)
        Phi = torch.stack([entry['fn'](self.R, self.Θ) for entry in self.basis], dim=2)
        # denominator S = 1 + Σ a_j φ_j
        S   = 1.0 - torch.tensordot(Phi, a_torch, dims=([2],[0]))
        integrand = Phi * (self.R.unsqueeze(2) / (S**2).unsqueeze(2))
        tmp = torch.trapz(integrand, dx=self.dθ, dim=1)  # integrate θ → (Nr, N)
        g_t = torch.trapz(tmp,       dx=self.dr, dim=0)  # integrate r → (N,)
        return g_t

    def residual(self, a, lam):
        """
        F(a,λ) = K·a − λ·g(a)
        a: numpy array (N,), lam: float
        returns numpy array (N,)
        """
        a_t = torch.tensor(a, dtype=self.Kt.dtype, device=self.Kt.device)
        g_t = self._compute_g_tensor(a_t)
        return self.K.dot(a) - lam * g_t.cpu().numpy()

    def jacobian(self, a, lam):
        """
        Auto‐differentiate F(a,λ) w.r.t [a, λ] using PyTorch.
        returns (F_a, F_lambda) as numpy arrays.
        """
        x0 = torch.tensor(np.concatenate([a, [lam]]),
                          dtype=self.Kt.dtype, device=self.Kt.device,
                          requires_grad=True)
        
        def F_torch(x):
            a_t   = x[:-1]
            lam_t = x[-1]
            return self.Kt.matmul(a_t) - lam_t*self._compute_g_tensor(a_t)

        J = torch.autograd.functional.jacobian(F_torch, x0)  # (N, N+1)
        J_np = J.detach().cpu().numpy()
        F_a    = J_np[:, :-1]
        F_lam  = J_np[:,  -1]
        return F_a, F_lam

    def compute_tangent(self, a, lam):
        """
        Solve [F_a  F_λ]·t = 0 for tangent direction,
        normalize, enforce t[-1]>0.
        """
        F_a, F_lam = self.jacobian(a, lam)
        A = np.hstack([F_a, F_lam.reshape(-1,1)])  # (N, N+1)
        _,_,Vt = np.linalg.svd(A)
        t = Vt.conj().T[:, -1]
        if t[0] < 0: t = -t
        return t / np.linalg.norm(t)

    def continue_(self, a0, lam0):
        """
        Run the pseudo‐arc‐length continuation.
        returns list of (a, lam).
        """
        path = []
        a, lam = a0.copy(), lam0
        t = self.compute_tangent(a, lam)

        for i in range(self.max_steps):
            if i % self.save_every == 0:
                print(f"Iteration {i}, λ = {lam:.6f}")
                
            x_prev = np.concatenate([a, [lam]])
            x_pred = x_prev + self.ds * t

            x = x_pred.copy()
            for newton_iter in range(20):
                a_k, lam_k = x[:-1], x[-1]
                Fvec = self.residual(a_k, lam_k)
                Φ    = t.dot(x - x_prev) - self.ds
                Faug = np.concatenate([Fvec, [Φ]])

                F_a, F_lam = self.jacobian(a_k, lam_k)
                Jaug = np.zeros((self.N+1, self.N+1))
                Jaug[:self.N, :self.N] = F_a
                Jaug[:self.N,   -1]    = F_lam
                Jaug[-1, :self.N]      = t[:-1]
                Jaug[-1,   -1]         = t[-1]

                delta = np.linalg.solve(Jaug, -Faug)
                x += delta
                if np.linalg.norm(delta) < self.newton_tol:
                    break

            a, lam = x[:-1], x[-1]
            t = self.compute_tangent(a, lam)
            if i % self.save_every == 0:
                path.append((a.copy(), lam))
            
        return path

    def reconstruct_shape(self, a):
        """
        Reconstruct the full shape using the coefficients a.
        Returns the displacement field u(r,θ) as a tensor.
        """
        # Initialize displacement field
        u = torch.zeros_like(self.R)
        
        # Add contribution from each basis function
        for i, entry in enumerate(self.basis):
            u += a[i] * entry['fn'](self.R, self.Θ)
            
        return u

    def get_max_displacement(self, a):
        """
        Get the maximum displxacement for a given set of coefficients.
        """
        u = self.reconstruct_shape(a)
        return torch.max(torch.abs(u)).item()

def main():
    # choose which zeros of J0/J1 to include
    zeros_list = [1, 2, 3]

    cont = PseudoArcLengthContinuation(
        zeros_list=zeros_list,
        ds=0.01,
        max_steps=200,
        newton_tol=1e-6,
        Nr=200,
        Nθ=400,
        save_every=10
    )

    # initial flat solution
    a0   = np.zeros(cont.N)
    lam0 = 0.0

    path = cont.continue_(a0, lam0)

    # Extract data for plotting
    lam_vals = [p[1] for p in path]
    max_disp = [cont.get_max_displacement(p[0]) for p in path]
    
    # Extract axisymmetric mode coefficients (every 3rd value)
    axisym_coeff = [[p[0][i] for i in range(0, len(p[0]), 3)] for p in path]
    
    # Calculate number of rows needed for subplots
    n_modes = len(axisym_coeff[0])
    n_rows = (n_modes + 1) // 2  # +1 for the max displacement plot
    
    # Create subplot grid
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5*n_rows))
    axes = axes.flatten()  # Flatten for easier indexing
    
    # Plot max displacement
    axes[0].plot(lam_vals, max_disp, 'o-')
    axes[0].set_xlabel('λ')
    axes[0].set_ylabel('Maximum Displacement')
    axes[0].set_title('Maximum Displacement vs λ')
    axes[0].grid(True)
    
    # Plot each axisymmetric mode coefficient
    for i in range(n_modes):
        mode_coeff = [coeff[i] for coeff in axisym_coeff]
        axes[i+1].plot(lam_vals, mode_coeff, 'o-')
        axes[i+1].set_xlabel('λ')
        axes[i+1].set_ylabel(f'a_{i*3}')
        axes[i+1].set_title(f'Axisymmetric Mode {i+1}')
        axes[i+1].grid(True)
    
    plt.tight_layout()
    plt.show()

    # Plot bifurcation diagram
    plt.figure(figsize=(10, 6))
    plt.plot(lam_vals, max_disp, 'o-')
    plt.xlabel("λ (electrostatic parameter)")
    plt.ylabel("Maximum Displacement")
    plt.title("Bifurcation Diagram")
    plt.grid(True)
    plt.show()

    # Plot the final shape
    final_a = path[-1][0]
    u = cont.reconstruct_shape(final_a)
    
    # Convert polar (R,Θ) to Cartesian (X,Y) coordinates
    X = cont.R.cpu().numpy() * np.cos(cont.Θ.cpu().numpy())
    Y = cont.R.cpu().numpy() * np.sin(cont.Θ.cpu().numpy())
    
    plt.figure(figsize=(8, 8))
    plt.contourf(X, Y, u.cpu().numpy(), levels=20)
    plt.colorbar(label='Displacement')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Final Membrane Shape')
    plt.axis('equal')  # Make axes equal scale
    plt.show()

if __name__ == "__main__":
    main()
