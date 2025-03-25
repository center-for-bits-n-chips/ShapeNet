import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import jn_zeros  
from sklearn.model_selection import train_test_split
from ShapeNet.ShapeNet import ShapeNet, LS_PINN_Loss
from torch.special import bessel_j0, bessel_j1
import plotly.graph_objects as go
import plotly.io as pio

n_basis = 2  # Number of zeros for basis functions
n_markers = 8 # Number of motion capture markers

data = np.loadtxt('data/ramp_data.csv', delimiter=',')  # Reads the CSV
data = data[::10] # reduce data by taking every 10th row
n_shapes = data.shape[0] # number of sampled shapes
shape_data = [] # training data to store

gap = 37.0 # mm
radius = 250.0 # mm

# radial coordinate of markers in mm
mocap_r = 1000.0 * torch.tensor([0.179333754, 0.12957065, 0.170481333, 0.167172452, 0.2115099, 0.009647519, 0.198512077, 0.236273679])
# angular coordinate of markers in radians
mocap_theta = torch.tensor([1.372922412, 0.945014831, 0.215990728, -1.167622835, -2.077150662, 1.32851777, 3.068736187, -2.969884485])
# normalize radius
mocap_r = mocap_r / radius

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
for i in range(n_shapes):
    voltage = data[i, 0]
    mocap_z = torch.tensor(data[i, 1:]*1/gap) # normalized
    mocap_z = - mocap_z # NOTE THE MINUS SIGN
    # Flatten the R_sparse and Theta_sparse to create a single row
    row = [i + 1, voltage]  # Start with the shape index
    for j in range(n_markers):
        row.append(mocap_r[j].item())      # Add sampled r
        row.append(mocap_theta[j].item())  # Add sampled θ
        row.append(mocap_z[j].item())      # Add sampled z

    # Generate basis functions for R_sparse
    basis_functions = generate_basis_functions(mocap_r, mocap_theta, n_basis)
    basis_functions_np = basis_functions.detach().numpy()  # Shape: (n_markers, n_basis)
    mocap_z_np = mocap_z.detach().numpy()  # Shape: (n_markers, 1)

    # Use least squares to find coefficients
    LS_coef, residuals, rank, s = np.linalg.lstsq(basis_functions_np, mocap_z_np, rcond=None)

    # Add the coefficients y_linear at the end
    row.extend(LS_coef.flatten().tolist())  # y_linear is shape (n_basis, 1), flatten to 1D

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
hidden_size = 128
output_size = 3*n_basis  # Number of basis function coefficients
lr = 0.001 # Learning rate

# Initialize the neural network
model = ShapeNet(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = LS_PINN_Loss(n_basis, bessel_zeros)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    losses.append(loss.item())
    test_losses.append(test_loss.item())

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

print('Alpha: ', criterion.alpha.item())
# loss
plt.figure(figsize=(8, 5))
plt.loglog(losses, label='Training Loss')
plt.loglog(test_losses, label='Test Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time, alpha = ' + str(criterion.alpha.item()))
plt.show()


# Bessel functions
def generate_basis_functions_for_plot(r, theta, N):
    basis_functions = []
    for k in range(N):
        n = k + 1 # indexing by n = 1, 2, 3
        phi_n = bessel_j0(bessel_zeros[k]*r)
        phi_n_sin = bessel_j1(bessel_zeros[n_basis + k]*r) * torch.sin(theta)
        phi_n_cos = bessel_j1(bessel_zeros[n_basis + k]*r) * torch.cos(theta)
        basis_functions.append(phi_n)
        basis_functions.append(phi_n_sin)
        basis_functions.append(phi_n_cos)
    return torch.stack(basis_functions, dim=2)  # Shape: (n_samples, n_basis)

# Choose a sample from the test set to visualize
for index in range(50,53):
    sample_X = X_test[index]
    sample_y = y_test[index]
    sample_q = voltage_test[index]  # Retrieve 'a' for the selected sample
    sample_q = float(sample_q)  # Ensure sample_a is a scalar

    R_plot_sparse = sample_X[1::3]
    Theta_plot_sparse = sample_X[2::3]
    Z_plot_sparse = sample_X[3::3]

    model.eval()
    # Use the model to predict the coefficients
    with torch.no_grad():
        predicted_coefficients = model(sample_X.unsqueeze(0)).numpy().flatten()
    # Generate paraboloid domain and mesh in cyllindrical coordinates
    r_full = torch.linspace(0, 1, 100)
    theta_full = torch.linspace(0, 2 * np.pi, 100)
    Theta, R = torch.meshgrid(theta_full, r_full, indexing='ij')
    # Generate full basis functions for plotting
    full_basis_functions = generate_basis_functions_for_plot(R, Theta, n_basis).detach().numpy()
    reconstructed_shape = np.dot(full_basis_functions, predicted_coefficients)

    # Original coefficients for comparison
    lsq_coefficients = sample_y.numpy()

    # occlude data
    # occlude_marker_index = 0
    # occluded_marker_x = sample_X[1 + 2*occlude_marker_index].clone().numpy()
    # occluded_marker_y = sample_X[1 + 2*occlude_marker_index + 1].clone().numpy()
    # X_sparse = sample_X[1::2].clone()
    # Y_sparse = sample_X[2::2].clone()
    # X_sparse[occlude_marker_index] = 0
    # Y_sparse[occlude_marker_index] = 0
    # basis_functions = generate_basis_functions(X_sparse.unsqueeze(1), n_basis)
    # basis_functions_np = basis_functions.detach().numpy()  # Shape: (n_samples, n_basis)
    # Y_sparse_np = Y_sparse.detach().numpy()  # Shape: (n_samples, 1)

    # model.eval()
    # # Use the model to predict the coefficients
    # with torch.no_grad():
    #     predicted_occluded_coefficients = model(sample_X.unsqueeze(0)).numpy().flatten()

    # reconstructed_shape_occluded = np.dot(full_basis_functions, predicted_occluded_coefficients)

    # Use least squares to find coefficients
    # lsq_occluded_coefficients, residuals, rank, s = np.linalg.lstsq(basis_functions_np, Y_sparse_np, rcond=None)

    # Ground truth shape using true coefficients
    lsq = np.dot(full_basis_functions, lsq_coefficients)
    # lsq_occluded = np.dot(full_basis_functions, lsq_occluded_coefficients)
    
    # Create bar graph
    x = np.arange(len(lsq_coefficients)) + 1  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/4, lsq_coefficients.flatten(), width, label='LSQ')
    rects2 = ax.bar(x + width/4, predicted_coefficients, width, label='NN')
    # rects3 = ax.bar(x + width, predicted_occluded_coefficients, width, label='NN occluded')
    plt.legend()
    ax.set_ylabel('Coefficient Value')
    ax.set_xticks(x)

    # X and Y for plotting
    X_plot = R * torch.cos(Theta)
    Y_plot = R * torch.sin(Theta)

    X_plot_sparse = R_plot_sparse * torch.cos(Theta_plot_sparse)
    Y_plot_sparse = R_plot_sparse * torch.sin(Theta_plot_sparse)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=45, azim=45, roll=0)

    # Plotting the noisy samples
    ax.scatter(radius*X_plot_sparse.numpy(), radius*Y_plot_sparse.numpy(), gap*Z_plot_sparse.numpy(), color='r', s=10)

    # Plot LSQ output
    #ax.plot_surface(X_plot.numpy(), Y_plot.numpy(), lsq, alpha=0.5, cmap='inferno')

    # Plot NN output
    ax.plot_surface(radius*X_plot.numpy(), radius*Y_plot.numpy(), gap*reconstructed_shape, alpha=0.5, cmap='plasma')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Ground Truth Paraboloid and Sampled Points')
    plt.show()

    #surface = go.Surface(x=X_plot.numpy(), y=Y_plot.numpy(), z=lsq, opacityscale=0.5)
    surface = go.Surface(x=radius*X_plot.numpy(), y=radius*Y_plot.numpy(), z=gap*reconstructed_shape, opacity = 0.5)
    sampled_points = go.Scatter3d(x=radius*X_plot_sparse.numpy(), y=radius*Y_plot_sparse.numpy(), z=gap*Z_plot_sparse.numpy(), mode='markers', marker=dict(size=5, color='red'))

    # Define the layout
    layout = go.Layout(
        title='3D Surface Plot',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            bgcolor='rgb(230,230,230)'
        )
    )

    # Create the figure
    fig = go.Figure(data=[surface, sampled_points], layout=layout)

    # Display the plot
    #fig.show()

model_filename = 'shape_net_model.pth'
# Save the model weights as before
torch.save(model.state_dict(), model_filename)

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