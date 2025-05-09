import numpy as np
import pandas as pd
import torch
#from torch.special import bessel_j0, bessel_j1
from scipy.special import j0, j1, jn_zeros
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.io as pio
import os
import sys
from shapenet_utils import generate_basis_functions_for_plot
sys.path.append('data analysis')
from read_dic import get_dic_data

def center_point_cloud(x, y, z, target_radius=250):
    """Center the point cloud and scale to target radius."""
    # Calculate centroid
    x_centroid = np.mean(x)
    y_centroid = np.mean(y)
    
    # Center the points
    x_centered = x - x_centroid
    y_centered = y - y_centroid
    
    return x_centered, y_centered, z

def cartesian_to_polar(x, y):
    """Convert Cartesian coordinates to polar coordinates."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def generate_basis_functions(r, theta, n_basis, bessel_zeros, radius):
    """Generate basis functions for shape fitting."""
    basis_functions = []
    for k in range(n_basis):
        n = k + 1 # indexing by n = 1, 2, 3
        phi_n = j0(bessel_zeros[k]*r/radius)
        phi_n_sin = j1(bessel_zeros[n_basis + k]*r/radius) * np.sin(theta)
        phi_n_cos = j1(bessel_zeros[n_basis + k]*r/radius) * np.cos(theta)
        basis_functions.append(phi_n)
        basis_functions.append(phi_n_sin)
        basis_functions.append(phi_n_cos)
    return basis_functions  # Shape: (n_samples, number of basis functions)

def fit_shape(x, y, z, n_basis=5, radius=250):
    """Fit the shape using basis functions."""
    # Center and scale the point cloud
    x_centered, y_centered, z = center_point_cloud(x, y, z)
    
    # Convert to polar coordinates
    r, theta = cartesian_to_polar(x_centered, y_centered)
    
    # Remove points outside radius
    # mask = r <= radius
    # x_centered = x_centered[mask]
    # y_centered = y_centered[mask]
    # z = z[mask]

    # r = r[mask]
    # theta = theta[mask]
    
    # Generate Bessel zeros (first n_basis zeros of J0 and J1)
    # Create Bessel Function Zeros Table
    bessel_zeros = []
    for m in range(2):
        zeros = jn_zeros(m, n_basis)
        bessel_zeros.extend(zeros)
    
    # Generate basis functions
    basis = generate_basis_functions(r, theta, n_basis, bessel_zeros, radius)
    
    # Solve least squares problem
    def residual(coefficients):
        return (np.dot(coefficients, np.array(basis)) - z)**2
    
    # Initial guess for coefficients
    initial_coefficients = np.zeros(3 * n_basis)
    
    # Solve least squares
    result = least_squares(residual, initial_coefficients)
    coefficients = result.x
    
    # Reconstruct shape
    reconstructed_z = np.dot(coefficients, np.array(basis))
    
    return x_centered, y_centered, reconstructed_z, coefficients

def plot_coefficients_over_time(time_s, all_coefficients, n_basis):
    """Plot the coefficients for each basis function over time."""
    fig, axes = plt.subplots(3, n_basis, figsize=(15, 10))
    fig.suptitle('Basis Function Coefficients Over Time')
    
    # Plot J0 coefficients
    for i in range(n_basis):
        axes[0, i].plot(time_s, all_coefficients[:, 3*i], 'b-')
        axes[0, i].set_title(f'J0 Basis {i+1}')
        axes[0, i].set_xlabel('Time (s)')
        axes[0, i].set_ylabel('Coefficient')
        axes[0, i].grid(True)
    
    # Plot J1*sin coefficients
    for i in range(n_basis):
        axes[1, i].plot(time_s, all_coefficients[:, 3*i+1], 'r-')
        axes[1, i].set_title(f'J1*sin Basis {i+1}')
        axes[1, i].set_xlabel('Time (s)')
        axes[1, i].set_ylabel('Coefficient')
        axes[1, i].grid(True)
    
    # Plot J1*cos coefficients
    for i in range(n_basis):
        axes[2, i].plot(time_s, all_coefficients[:, 3*i+2], 'g-')
        axes[2, i].set_title(f'J1*cos Basis {i+1}')
        axes[2, i].set_xlabel('Time (s)')
        axes[2, i].set_ylabel('Coefficient')
        axes[2, i].grid(True)
    
    plt.tight_layout()
    plt.show()

def create_shape_animation(x_centered, y_centered, z, coefficients, n_basis=5, radius=250):
    """
    Create an animation of point cloud data and fitted surface over time.
    
    Args:
        x_centered: List of x coordinates for each frame
        y_centered: List of y coordinates for each frame
        z: List of z coordinates for each frame
        coefficients: Array of coefficients for each frame
        n_basis: Number of basis functions to use
    """
    # Create directory for animations if it doesn't exist
    os.makedirs("animations", exist_ok=True)

    # Generate Bessel zeros
    bessel_zeros = []
    for m in range(2):
        zeros = jn_zeros(m, n_basis)
        bessel_zeros.extend(zeros)

    # Generate mesh for plotting
    r_full = torch.linspace(0, 1, 100)
    theta_full = torch.linspace(0, 2 * np.pi, 100)
    Theta, R = torch.meshgrid(theta_full, r_full, indexing='ij')
    X = radius * R * torch.cos(Theta)
    Y = radius * R * torch.sin(Theta)
    
    # Generate full basis functions for plotting
    full_basis_functions = generate_basis_functions_for_plot(R, Theta, n_basis, bessel_zeros).detach().numpy()
    
    # Create frames for animation
    frames = []
    
    # Create frames
    for frame_idx, frame_coeffs in enumerate(coefficients):
        # Calculate Z values using basis functions and coefficients
        Z = np.dot(full_basis_functions, frame_coeffs)
        
        # Create frame
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=x_centered[frame_idx],
                    y=y_centered[frame_idx],
                    z=z[frame_idx],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=z[frame_idx],
                        colorscale='Viridis',
                        opacity=0.8,
                        cmin=-30,
                        cmax=0
                    ),
                    name='Point Cloud'
                ),
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    colorscale='Viridis',
                    opacity=0.8,
                    showscale=True,  # Show colorbar on every frame
                    cmin=-30,  # Fix color scale minimum
                    cmax=0,    # Fix color scale maximum
                    name='Reconstructed Surface'
                )
            ],
            name=f"Frame {frame_idx}"
        )
        frames.append(frame)
    
    # Create initial figure
    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )
    
     # Add animation controls
    fig.update_layout(
        title="ShapeNet Neural Network Prediction Animation",
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            zaxis=dict(range=[-30, 5]),  # Set z limits from -30 to 0
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 200, "redraw": True},
                                       "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Voltage: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 200},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [f.name],
                        {
                            "frame": {"duration": 200, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 200}
                        }
                    ],
                    "label": f.name,
                    "method": "animate"
                }
                for f in frames
            ]
        }]
    )
    
    # Save the animation as HTML
    output_path = "animations/shape_reconstruction_animation.html"
    pio.write_html(fig, file=output_path, auto_open=False)
    print(f"Saved interactive Plotly animation to '{output_path}'")
    
    return output_path
def main():
    # Directory containing the DIC data
    directory = '28 mm closed loop take 1'
    
    # Read DIC data
    print("Reading DIC data files...")
    data_frames, time_s, voltage_kV = get_dic_data(directory)
    print(f"Successfully read {len(data_frames)} frames")
    
    # Initialize arrays to store coefficients and errors
    n_basis = 5
    all_coefficients = np.zeros((len(data_frames), 3 * n_basis))
    all_mse = np.zeros(len(data_frames))
    
    # Store centered coordinates for animation
    x_centered_list = []
    y_centered_list = []
    z_list = []

    radius = 250
    
    # Process each frame
    for frame_idx, df in enumerate(data_frames):
        print(f"\nProcessing frame {frame_idx}")
        
        # Get coordinates
        x = df['x[mm]'] + df['x-displacement[mm]']
        y = df['y[mm]'] + df['y-displacement[mm]']
        z = df['z[mm]'] + df['z-displacement[mm]']
        
        # Fit shape
        x_centered, y_centered, reconstructed_z, coefficients = fit_shape(x, y, z, n_basis, radius)
        
        # Store results
        all_coefficients[frame_idx] = coefficients
        all_mse[frame_idx] = np.sqrt(np.mean((z - reconstructed_z)**2))
        
        # Store coordinates for animation
        x_centered_list.append(x_centered)
        y_centered_list.append(y_centered)
        z_list.append(z)
        
        # Print fitting statistics
        print(f"Mean Squared Error: {all_mse[frame_idx]:.4f}")
        print(f"Coefficients: {coefficients}")
    
    # Create animation
    create_shape_animation(x_centered_list, y_centered_list, z_list, all_coefficients, n_basis, radius)
    
    # Plot coefficients over time
    plot_coefficients_over_time(time_s, all_coefficients, n_basis)
    
    # Plot mean squared error over time
    plt.figure(figsize=(10, 5))
    plt.plot(time_s, all_mse, 'b-')
    plt.xlabel('Time (s)')
    plt.ylabel('Root Mean Squared Error (mm)')
    plt.title('Fitting Error Over Time')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
