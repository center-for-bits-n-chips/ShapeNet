import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import jn_zeros  
from ShapeNet.ShapeNet import ShapeNet, LS_PINN_Loss
from torch.special import bessel_j0, bessel_j1
from sklearn.model_selection import train_test_split
from read_dat import read_labview_binary, cartesian_to_cylindrical
import itertools
import random  # Add this import for random sampling

n_basis = 3  # Number of zeros for basis functions
n_markers = 15 # Number of motion capture markers

# Parameters
gap = 37.0 # mm
radius = 250.0 # mm

# Load data from .dat file instead of CSV
print("Reading data file...")
dat_filename = "data/2025-05-26 28 mm stabilized pull-in.dat"
voltages, positions, time = read_labview_binary(dat_filename, decimate=100)  # Using decimate=10 to reduce data similar to original
mesh_voltage = voltages[:,0] - voltages[:,1]

# Calculate cylindrical coordinates
r, theta, z = cartesian_to_cylindrical(positions)

# Create mocap tensor values
mocap_r = torch.tensor(r[0])  # Use first frame's r values
mocap_theta = torch.tensor(theta[0])  # Use first frame's theta values
# Normalize radius
mocap_r = mocap_r / radius

n_shapes = len(mesh_voltage)  # number of sampled shapes
shape_data = []  # training data to store

# Create Bessel Function Zeros Table
bessel_zeros = []
for m in range(2):
    zeros = jn_zeros(m, n_basis)
    bessel_zeros.extend(zeros)

# Bessel functions
def generate_basis_functions(r, theta, N):
    basis_functions = []
    for k in range(N):
        n = k + 1 # indexing by n = 1, 2, 3
        phi_n = bessel_j0(bessel_zeros[k]*r)
        phi_n_sin = bessel_j1(bessel_zeros[n_basis + k]*r) * torch.sin(theta)
        phi_n_cos = bessel_j1(bessel_zeros[n_basis + k]*r) * torch.cos(theta)
        basis_functions.append(phi_n)
        basis_functions.append(phi_n_sin)
        basis_functions.append(phi_n_cos)
    return torch.stack(basis_functions, dim=1)  # Shape: (n_markers, n_basis)

# Generate shapes data table
n_permutations = 15  # Number of random permutations to generate per shape
for i in range(n_shapes):
    volt = mesh_voltage[i]
    mocap_z = torch.tensor(z[i] * 1/gap)  # normalized
    mocap_z = - mocap_z  # NOTE THE MINUS SIGN
    
    # Create marker data as a list of tuples (r, theta, z)
    marker_data = [(mocap_r[j].item(), mocap_theta[j].item(), mocap_z[j].item()) for j in range(n_markers)]
    
    # First add the original ordering
    row = [i + 1, volt]
    for marker in marker_data:
        row.extend(marker)
    
    # Generate basis functions for original data
    basis_functions = generate_basis_functions(mocap_r, mocap_theta, n_basis)
    basis_functions_np = basis_functions.detach().numpy()
    mocap_z_np = mocap_z.detach().numpy().reshape(-1, 1)
    
    # Use least squares to find coefficients
    LS_coef, residuals, rank, s = np.linalg.lstsq(basis_functions_np, mocap_z_np, rcond=None)
    row.extend(LS_coef.flatten().tolist())
    shape_data.append(row)
    
    # Generate random permutations
    for _ in range(n_permutations):
        # Create a random permutation of the marker data
        perm = random.sample(marker_data, len(marker_data))
        
        # Start with the shape index and voltage
        row = [i + 1, volt]
        
        # Flatten the permuted marker data
        for marker in perm:
            row.extend(marker)
        
        # Generate basis functions for the permuted R and Theta values
        perm_r = torch.tensor([m[0] for m in perm])
        perm_theta = torch.tensor([m[1] for m in perm])
        perm_z = torch.tensor([m[2] for m in perm])
        
        # Generate basis functions for the permuted data
        basis_functions = generate_basis_functions(perm_r, perm_theta, n_basis)
        basis_functions_np = basis_functions.detach().numpy()
        perm_z_np = perm_z.detach().numpy().reshape(-1, 1)
        
        # Use least squares to find coefficients
        LS_coef, residuals, rank, s = np.linalg.lstsq(basis_functions_np, perm_z_np, rcond=None)
        
        # Add the coefficients
        row.extend(LS_coef.flatten().tolist())
        
        shape_data.append(row)

# Create column names
columns = ['n', 'v']
for i in range(1, n_markers + 1):
    columns.append(f'r_{i}')
    columns.append(f'theta_{i}')
    columns.append(f'z_{i}')
for i in range(1, 3*n_basis + 1):
    columns.append(f'LS_coef_{i}')

# Create a Pandas DataFrame
shape_df = pd.DataFrame(shape_data, columns=columns)
print(shape_df.head())  # Print the first few rows of the DataFrame

## X is the mocap
## y is the LS
# Prepare X and y for training
X = shape_df.drop(columns=['n'] + [f'LS_coef_{i}' for i in range(1, 3*n_basis + 1)])
X = X.values.astype(np.float32)

y = shape_df[columns[-3*n_basis:]]  # Coefficient columns as target
y = y.values.astype(np.float32)

# Train Test Split
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

voltage_train = X_train[:, 0]  # Assuming 'q' is the second column after 'n' is dropped
voltage_test = X_test[:, 0]

X_train = torch.FloatTensor(X_train)  # Convert to float tensor for PyTorch
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# Define Model & Training Parameters
input_size = X_train.shape[1]  # Number of features in X_train
output_size = 3*n_basis  # Number of basis function coefficients
hidden_size = 32 #(input_size*output_size)**0.5
lr = 0.001 # Learning rate

# Initialize the neural network
model = ShapeNet(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = LS_PINN_Loss(n_basis, bessel_zeros)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       factor=0.5, patience=125)

# Training loop
num_epochs = 10000
losses = []
test_losses = []
for epoch in range(num_epochs):
    model.train()
    # Forward pass
    outputs = model(X_train)

    loss = criterion(outputs, X_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, X_test)
    scheduler.step(test_loss)

    losses.append(loss.item())
    test_losses.append(test_loss.item())

    # if scheduler.num_bad_epochs > 100:
    #     print("Early stop at epoch", epoch)
    #     break

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

print('Alpha: ', criterion.alpha.item())

# Plot training loss
plt.figure(figsize=(8, 5))
plt.loglog(losses, label='Training Loss')
plt.loglog(test_losses, label='Test Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time, alpha = ' + str(criterion.alpha.item()))
plt.show()

# Save the model weights
model_filename = 'shape_net_model.pth'
torch.save(model.state_dict(), model_filename)
print(f"Model saved to {model_filename}")

# Benchmark model inference speed
import torch.utils.benchmark as benchmark

num_iterations = 100  # Number of times to repeat the statement

def forward_pass():
    with torch.no_grad():
        _ = model(X_test[0])

timer = benchmark.Timer(
    stmt='forward_pass()',
    setup='from __main__ import forward_pass',
    num_threads=torch.get_num_threads()
)

result = timer.timeit(num_iterations)
print(f"Average time per forward pass: {result.mean * 1000:.3f} ms")