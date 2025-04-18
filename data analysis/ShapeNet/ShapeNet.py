import torch
import torch.nn as nn
from torch.special import bessel_j0, bessel_j1

# Define the neural network model
class ShapeNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ShapeNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size) # Output layer
        self.dropout = nn.Dropout(p=0.2)  # Dropout with 20% probability

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out
    
class LS_PINN_Loss(nn.Module):
    def __init__(self, n_basis, bessel_zeros, initial_alpha = 0.0001):
        super(LS_PINN_Loss, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        self.n_basis = n_basis
        self.bessel_zeros = bessel_zeros

    def generate_basis_functions(self, r, theta, N):
        basis_functions = []
        for k in range(N):
            n = k + 1 # indexing by n = 1, 2, 3
            phi_n = bessel_j0(self.bessel_zeros[k]*r)
            phi_n_sin = bessel_j1(self.bessel_zeros[self.n_basis + k]*r) * torch.sin(theta)
            phi_n_cos = bessel_j1(self.bessel_zeros[self.n_basis + k]*r) * torch.cos(theta)
            basis_functions.append(phi_n)
            basis_functions.append(phi_n_sin)
            basis_functions.append(phi_n_cos)
        return torch.stack(basis_functions, dim=2)  # Shape: (batch size, n_samples, number of basis functions)

    def forward(self, outputs, inputs):
        voltage = inputs[:, 0]
        mocap_r = inputs[:, 1::3]
        mocap_theta = inputs[:, 2::3]
        mocap_z = inputs[:, 3::3]

        basis_functions = self.generate_basis_functions(mocap_r, mocap_theta, self.n_basis) # (batch size, sample size, basis size)
        reconstructed_shape = torch.einsum('ijk,ik->ij', basis_functions, outputs)

        LS_loss = ((reconstructed_shape - mocap_z)**2).mean()

        #eigenvalues = generate_eigenvalues(n_basis, m_basis) # (basis size)
        #modal_force_coefficients = generate_modal_force_coefficients(n_basis)

        #PDE_loss = self.alpha * ((0.5 * outputs * eigenvalues - modal_force_coefficients.unsqueeze(0) * q.unsqueeze(1))**2).mean()

        return LS_loss