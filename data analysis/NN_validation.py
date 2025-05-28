import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import os
import plotly.graph_objects as go
import plotly.io as pio

from read_dat import read_labview_binary, cartesian_to_cylindrical
from ShapeNet.ShapeNet import ShapeNet
from shapenet_utils import (
    create_bessel_zeros_table,
    generate_basis_functions,
    generate_basis_functions_for_plot,
    prepare_data_from_shape_df,
    create_matplotlib_animation,
    create_plotly_animation
)

def load_and_prepare_data(dat_filename, n_basis, n_markers, radius, gap, decimate=1):
    """
    Load data from .dat file and prepare for processing.
    
    Args:
        dat_filename: Path to the .dat file
        n_basis: Number of zeros for basis functions
        n_markers: Number of motion capture markers
        radius: Physical radius for scaling (mm)
        gap: Physical gap for scaling (mm)
        decimate: Decimation factor for data reading
        
    Returns:
        shape_df: DataFrame containing shape data
        bessel_zeros: List of Bessel function zeros
    """
    print("Reading data file...")
    voltages, positions, time = read_labview_binary(dat_filename, decimate=decimate)
    mesh_voltage = voltages[:,0] - voltages[:,1]

    # Calculate cylindrical coordinates
    r, theta, z = cartesian_to_cylindrical(positions)

    # Create mocap tensor values
    mocap_r = torch.tensor(r[0])  # Use first frame's r values
    mocap_theta = torch.tensor(theta[0])  # Use first frame's theta values
    # Normalize radius
    mocap_r = mocap_r / radius

    # Create Bessel Function Zeros Table
    bessel_zeros = create_bessel_zeros_table(n_basis)

    n_shapes = len(mesh_voltage)  # number of sampled shapes
    shape_data = []  # data to store

    # Generate shapes data table with LSQ coefficients
    for i in range(n_shapes):
        volt = mesh_voltage[i]
        mocap_z = torch.tensor(z[i] * 1/gap)  # normalized
        mocap_z = -mocap_z  # NOTE THE MINUS SIGN
        
        # Flatten the R_sparse and Theta_sparse to create a single row
        row = [i + 1, volt]  # Start with the shape index
        for j in range(n_markers):
            row.append(mocap_r[j].item())      # Add sampled r
            row.append(mocap_theta[j].item())  # Add sampled θ
            row.append(mocap_z[j].item())      # Add sampled z

        # Generate basis functions for R_sparse
        basis_functions = generate_basis_functions(mocap_r, mocap_theta, n_basis, bessel_zeros)
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
    
    return shape_df, bessel_zeros

def load_model(model_filename, input_size, hidden_size, output_size):
    """
    Load the neural network model.
    
    Args:
        model_filename: Path to the model file
        input_size: Number of input features
        hidden_size: Number of hidden units
        output_size: Number of output features
        
    Returns:
        model: Loaded ShapeNet model
    """
    model = ShapeNet(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_filename))
    model.eval()  # Set model to evaluation mode
    return model

def visualize_coefficient_comparison(sample_X, sample_y, predicted_coefficients):
    """
    Create a bar graph comparing true coefficients with predicted ones and save to file.
    
    Args:
        sample_X: Input sample tensor
        sample_y: Target coefficient tensor
        predicted_coefficients: Model predicted coefficients
        
    Returns:
        Path to the saved bargraph image
    """
    lsq_coefficients = sample_y.numpy()
    sample_q = sample_X[0].item()  # Get voltage value
    
    # Create bar graph comparing coefficients
    x = np.arange(len(lsq_coefficients)) + 1  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/4, lsq_coefficients.flatten(), width, label='LSQ')
    rects2 = ax.bar(x + width/4, predicted_coefficients, width, label='NN')
    ax.set_title(f'Coefficient Comparison at Voltage = {sample_q:.2f}')
    ax.set_ylabel('Coefficient Value')
    ax.set_xlabel('Coefficient Index')
    ax.set_xticks(x)
    ax.legend()
    plt.tight_layout()
    
    # Create directory for bargraphs if it doesn't exist
    os.makedirs("bargraphs", exist_ok=True)
    
    # Save the figure instead of showing it
    filename = f"bargraphs/coefficients_{sample_q:.2f}.png"
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved coefficient comparison to {filename}")
    
    return filename

def visualize_3d_surface(sample_X, sample_y, predicted_coefficients, bessel_zeros, n_basis, radius, gap):
    """
    Create a 3D visualization of the predicted surface using Plotly.
    
    Args:
        sample_X: Input sample tensor
        sample_y: Target coefficient tensor
        predicted_coefficients: Model predicted coefficients
        bessel_zeros: List of Bessel function zeros
        n_basis: Number of basis functions
        radius: Physical radius for scaling (mm)
        gap: Physical gap for scaling (mm)
        
    Returns:
        Path to the saved HTML file
    """
    sample_q = sample_X[0].item()  # Get voltage value

    # Extract coordinates from sample_X
    R_plot_sparse = sample_X[1::3]
    Theta_plot_sparse = sample_X[2::3]
    Z_plot_sparse = sample_X[3::3]

    # Generate mesh for plotting
    r_full = torch.linspace(0, 1, 100)
    theta_full = torch.linspace(0, 2 * np.pi, 100)
    Theta, R = torch.meshgrid(theta_full, r_full, indexing='ij')
    
    # Generate full basis functions for plotting
    full_basis_functions = generate_basis_functions_for_plot(R, Theta, n_basis, bessel_zeros).detach().numpy()
    reconstructed_shape = np.dot(full_basis_functions, predicted_coefficients)

    # Calculate X and Y for 3D plotting
    X_plot = R * torch.cos(Theta)
    Y_plot = R * torch.sin(Theta)

    X_plot_sparse = R_plot_sparse * torch.cos(Theta_plot_sparse)
    Y_plot_sparse = R_plot_sparse * torch.sin(Theta_plot_sparse)

    # Create Plotly interactive visualization
    surface = go.Surface(
        x=radius*X_plot.numpy(), 
        y=radius*Y_plot.numpy(), 
        z=gap*reconstructed_shape, 
        opacity=0.5,
        colorscale='Viridis',
        name='NN Prediction'
    )
    
    sampled_points = go.Scatter3d(
        x=radius*X_plot_sparse.numpy(), 
        y=radius*Y_plot_sparse.numpy(), 
        z=gap*Z_plot_sparse.numpy(), 
        mode='markers', 
        marker=dict(size=5, color='red'),
        name='Data Points'
    )

    # Define the layout
    layout = go.Layout(
        title=f'3D Surface Plot at Voltage = {sample_q:.2f}',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            bgcolor='rgb(230,230,230)'
        )
    )

    # Create the figure
    fig = go.Figure(data=[surface, sampled_points], layout=layout)
    
    # Create directory for surface plots if it doesn't exist
    os.makedirs("surface_plots", exist_ok=True)
    
    # Save the plot as an HTML file
    filename = f"surface_plots/plot_voltage_{sample_q:.2f}.html"
    fig.write_html(filename)
    print(f"Saved interactive plot to {filename}")
    
    return filename

def validate_and_visualize_samples(model, X_tensor, y_tensor, bessel_zeros, n_basis, radius, gap, num_samples=8):
    """
    Validate model on samples and create visualizations.
    
    Args:
        model: ShapeNet model
        X_tensor: Input data tensor
        y_tensor: Target data tensor
        bessel_zeros: List of Bessel function zeros
        n_basis: Number of basis functions
        radius: Physical radius for scaling (mm)
        gap: Physical gap for scaling (mm)
        num_samples: Number of samples to visualize
        
    Returns:
        List of created files
    """
    created_files = []
    for index in range(len(X_tensor)):
        sample_X = X_tensor[index]
        sample_y = y_tensor[index]
        sample_q = sample_X[0].item()  # Get voltage value

        print(f"\nProcessing sample {index+1}/{len(X_tensor)} (Voltage: {sample_q:.2f}V)...")

        # Use the model to predict the coefficients
        with torch.no_grad():
            predicted_coefficients = model(sample_X.unsqueeze(0)).numpy().flatten()
        
        # Compute mean squared error
        lsq_coefficients = sample_y.numpy()
        mse = np.mean((lsq_coefficients - predicted_coefficients) ** 2)
        print(f"  Mean Squared Error: {mse:.6f}")
        
        # Visualize coefficient comparison
        bargraph_file = visualize_coefficient_comparison(sample_X, sample_y, predicted_coefficients)
        created_files.append(bargraph_file)
        
        # Visualize 3D surface
        surface_file = visualize_3d_surface(sample_X, sample_y, predicted_coefficients, bessel_zeros, n_basis, radius, gap)
        created_files.append(surface_file)
        
    return created_files

def create_animations(model, X_tensor, y_tensor, bessel_zeros, n_basis, radius, gap):
    """
    Create animations of the model predictions.
    
    Args:
        model: ShapeNet model
        X_tensor: Input data tensor
        y_tensor: Target data tensor
        bessel_zeros: List of Bessel function zeros
        n_basis: Number of basis functions
        radius: Physical radius for scaling (mm)
        gap: Physical gap for scaling (mm)
        
    Returns:
        List of created animation files
    """
    print("Creating animations...")
    animation_files = []

    # Create directory for animations if it doesn't exist
    os.makedirs("animations", exist_ok=True)

    # Create animations
    try:
        matplotlib_file = create_matplotlib_animation(model, X_tensor, y_tensor, n_basis, bessel_zeros, radius, gap)
        animation_files.append(matplotlib_file)
    except Exception as e:
        print(f"Error creating matplotlib animation: {e}")
        print("Make sure ffmpeg is installed for saving matplotlib animations")

    try:
        plotly_file = create_plotly_animation(model, X_tensor, y_tensor, n_basis, bessel_zeros, radius, gap)
        animation_files.append(plotly_file)
    except Exception as e:
        print(f"Error creating Plotly animation: {e}")

    print(f"Animation generation complete! Created {len(animation_files)} animation files.")
    return animation_files

def plot_max_displacement_vs_voltage(model, X_tensor, y_tensor, bessel_zeros, n_basis, radius, gap):
    """
    Create a plot comparing maximum displacement vs voltage for both NN predictions and actual data.
    
    Args:
        model: ShapeNet model
        X_tensor: Input data tensor
        y_tensor: Target data tensor
        bessel_zeros: List of Bessel function zeros
        n_basis: Number of basis functions
        radius: Physical radius for scaling (mm)
        gap: Physical gap for scaling (mm)
        
    Returns:
        Path to the saved plot image
    """
    # Generate mesh for plotting
    r_full = torch.linspace(0, 1, 100)
    theta_full = torch.linspace(0, 2 * np.pi, 100)
    Theta, R = torch.meshgrid(theta_full, r_full, indexing='ij')
    
    # Generate full basis functions for plotting
    full_basis_functions = generate_basis_functions_for_plot(R, Theta, n_basis, bessel_zeros).detach().numpy()
    
    # Initialize lists to store results
    voltages = []
    max_displacements_nn = []
    max_displacements_actual = []
    
    # Process each sample
    for idx in range(len(X_tensor)):
        sample_X = X_tensor[idx]
        sample_y = y_tensor[idx]
        voltage = sample_X[0].item()
        voltages.append(voltage)
        
        # Get actual coefficients and calculate max displacement
        actual_coefficients = sample_y.numpy()
        actual_shape = np.dot(full_basis_functions, actual_coefficients)
        max_displacements_actual.append(np.max(np.abs(gap * actual_shape)))
        
        # Get NN predictions and calculate max displacement
        with torch.no_grad():
            predicted_coefficients = model(sample_X.unsqueeze(0)).numpy().flatten()
        predicted_shape = np.dot(full_basis_functions, predicted_coefficients)
        max_displacements_nn.append(np.max(np.abs(gap * predicted_shape)))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(voltages, max_displacements_actual, 'b.-', label='Actual Data', alpha=0.7)
    plt.plot(voltages, max_displacements_nn, 'r.-', label='NN Prediction', alpha=0.7)
    
    plt.title('Maximum Displacement vs Voltage')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Maximum Displacement (mm)')
    plt.grid(True)
    plt.legend()
    
    # Create directory for plots if it doesn't exist
    os.makedirs("displacement_plots", exist_ok=True)
    
    # Save the plot
    filename = "displacement_plots/max_displacement_vs_voltage.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved max displacement vs voltage plot to {filename}")
    
    return filename

def plot_coefficients_vs_voltage(model, X_tensor, y_tensor):
    """
    Create a plot showing how each coefficient changes with voltage.
    
    Args:
        model: ShapeNet model
        X_tensor: Input data tensor
        y_tensor: Target data tensor
        
    Returns:
        Path to the saved plot image
    """
    # Initialize lists to store results
    voltages = []
    predicted_coefficients = []
    actual_coefficients = []
    
    # Process each sample
    # for idx in range(len(X_tensor)):
    for idx in range(50, 2001, 200):
        sample_X = X_tensor[idx]
        sample_y = y_tensor[idx]
        voltage = sample_X[0].item()
        voltages.append(voltage)
        
        # Get actual coefficients
        actual_coefficients.append(sample_y.numpy().flatten())
        # Get NN predictions
        with torch.no_grad():
            predicted = model(sample_X.unsqueeze(0)).numpy().flatten()
            predicted_coefficients.append(predicted)
    
    # Convert to numpy arrays
    predicted_coefficients = np.array(predicted_coefficients)
    actual_coefficients = np.array(actual_coefficients)
    
    # Create subplots for each coefficient
    n_coefficients = predicted_coefficients.shape[1]
    n_cols = 2
    n_rows = (n_coefficients + 1) // 2
    
    plt.figure(figsize=(12, 4*n_rows))
    
    for i in range(n_coefficients):
        plt.subplot(n_rows, n_cols, i+1)
        plt.plot(voltages, actual_coefficients[:, i], 'b.-', label='Actual', alpha=0.7)
        plt.plot(voltages, predicted_coefficients[:, i], 'r.-', label='NN Prediction', alpha=0.7)
        plt.title(f'Coefficient {i+1} vs Voltage')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Coefficient Value')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    
    # Create directory for plots if it doesn't exist
    os.makedirs("coefficient_plots", exist_ok=True)
    
    # Save the plot
    filename = "coefficient_plots/coefficients_vs_voltage.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved coefficients vs voltage plot to {filename}")
    
    return filename

def main():
    """Main function to run validation and visualization."""
    print("Starting ShapeNet validation and visualization...")
    
    # Parameters - must match the training parameters
    n_basis = 3  # Number of zeros for basis functions
    n_markers = 15  # Number of motion capture markers
    gap = 37.0  # mm
    radius = 250.0  # mm
    dat_filename = "data/2025-05-27 case A.dat"
    model_filename = 'shape_net_model.pth'
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    shape_df, bessel_zeros = load_and_prepare_data(
        dat_filename, n_basis, n_markers, radius, gap, decimate=10
    )
    
    # Prepare X and y for validation
    print("\n2. Preparing tensors for validation...")
    X_tensor, y_tensor = prepare_data_from_shape_df(shape_df, n_basis)
    print(f"   Input tensor shape: {X_tensor.shape}")
    print(f"   Target tensor shape: {y_tensor.shape}")
    
    # Define Model parameters
    input_size = X_tensor.shape[1]  # Number of features in X_tensor
    output_size = 3*n_basis  # Number of basis function coefficients
    hidden_size = 32
    
    # Load the neural network model
    print("\n3. Loading neural network model...")
    model = load_model(model_filename, input_size, hidden_size, output_size)
    print(f"   Model loaded from: {model_filename}")
    
    # Validate and visualize samples
    # print("\n4. Validating and visualizing samples...")
    # created_files = validate_and_visualize_samples(model, X_tensor, y_tensor, bessel_zeros, n_basis, radius, gap)
    # print(f"   Created {len(created_files)} visualization files")
    
    # Create animations
    print("\n5. Creating animations...")
    #animation_files = create_animations(model, X_tensor, y_tensor, bessel_zeros, n_basis, radius, gap)
    
    # Plot maximum displacement vs voltage
    print("\n6. Plotting maximum displacement vs voltage...")
    max_displacement_file = plot_max_displacement_vs_voltage(model, X_tensor, y_tensor, bessel_zeros, n_basis, radius, gap)
    
    # Plot coefficients vs voltage
    print("\n7. Plotting coefficients vs voltage...")
    #coefficients_file = plot_coefficients_vs_voltage(model, X_tensor, y_tensor)
    
    print("\nValidation and visualization complete!")
    # print(f"Created {len(created_files)} visualization files and animations.")

if __name__ == "__main__":
    main() 