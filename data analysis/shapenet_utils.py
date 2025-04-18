import torch
import numpy as np
from scipy.special import jn_zeros
from torch.special import bessel_j0, bessel_j1

# Bessel functions for generating basis functions
def generate_basis_functions(r, theta, n_basis, bessel_zeros):
    """
    Generate basis functions for shape representation.
    
    Args:
        r: Normalized radial coordinates
        theta: Angular coordinates
        n_basis: Number of zeros for basis functions
        bessel_zeros: List of Bessel function zeros
        
    Returns:
        Tensor of basis functions with shape (n_markers, n_basis)
    """
    basis_functions = []
    for k in range(n_basis):
        n = k + 1  # indexing by n = 1, 2, 3
        phi_n = bessel_j0(bessel_zeros[k]*r)
        phi_n_sin = bessel_j1(bessel_zeros[n_basis + k]*r) * torch.sin(theta)
        phi_n_cos = bessel_j1(bessel_zeros[n_basis + k]*r) * torch.cos(theta)
        basis_functions.append(phi_n)
        basis_functions.append(phi_n_sin)
        basis_functions.append(phi_n_cos)
    return torch.stack(basis_functions, dim=1)  # Shape: (n_markers, n_basis)

# Bessel functions for plotting
def generate_basis_functions_for_plot(r, theta, n_basis, bessel_zeros):
    """
    Generate basis functions for plotting.
    
    Args:
        r: Normalized radial coordinates
        theta: Angular coordinates
        n_basis: Number of zeros for basis functions
        bessel_zeros: List of Bessel function zeros
        
    Returns:
        Tensor of basis functions with shape (n_samples, n_basis)
    """
    basis_functions = []
    for k in range(n_basis):
        n = k + 1  # indexing by n = 1, 2, 3
        phi_n = bessel_j0(bessel_zeros[k]*r)
        phi_n_sin = bessel_j1(bessel_zeros[n_basis + k]*r) * torch.sin(theta)
        phi_n_cos = bessel_j1(bessel_zeros[n_basis + k]*r) * torch.cos(theta)
        basis_functions.append(phi_n)
        basis_functions.append(phi_n_sin)
        basis_functions.append(phi_n_cos)
    return torch.stack(basis_functions, dim=2)  # Shape: (n_samples, n_samples, n_basis)

def create_bessel_zeros_table(n_basis):
    """
    Create a table of Bessel function zeros.
    
    Args:
        n_basis: Number of zeros for basis functions
        
    Returns:
        List of Bessel function zeros
    """
    bessel_zeros = []
    for m in range(2):
        zeros = jn_zeros(m, n_basis)
        bessel_zeros.extend(zeros)
    return bessel_zeros

def prepare_data_from_shape_df(shape_df, n_basis):
    """
    Prepare input and target tensors from shape DataFrame.
    
    Args:
        shape_df: DataFrame containing shape data
        n_basis: Number of zeros for basis functions
        
    Returns:
        X_tensor: Input tensor for model
        y_tensor: Target tensor for model
    """
    # Drop 'n' column and coefficient columns from X
    X = shape_df.drop(columns=['n'] + [f'LS_coef_{i}' for i in range(1, 3*n_basis + 1)])
    X = X.values.astype(np.float32)

    # Get coefficient columns as target
    columns = shape_df.columns.tolist()
    y = shape_df[columns[-3*n_basis:]]  # Coefficient columns as target
    y = y.values.astype(np.float32)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    return X_tensor, y_tensor

def create_matplotlib_animation(model, X_tensor, y_tensor, n_basis, bessel_zeros, radius, gap):
    """
    Creates a matplotlib-based animation of the model predictions.
    
    Args:
        model: ShapeNet model
        X_tensor: Input data tensor
        y_tensor: Target data tensor
        n_basis: Number of basis functions
        bessel_zeros: List of Bessel function zeros
        radius: Physical radius for scaling
        gap: Physical gap for scaling
        
    Returns:
        Path to the saved animation file
    """
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    import matplotlib.animation as animation
    
    # Create directory for animations if it doesn't exist
    os.makedirs("animations", exist_ok=True)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=45, roll=0)
    
    # Generate mesh for plotting (only once)
    r_full = torch.linspace(0, 1, 100)
    theta_full = torch.linspace(0, 2 * np.pi, 100)
    Theta, R = torch.meshgrid(theta_full, r_full, indexing='ij')
    X_plot = R * torch.cos(Theta)
    Y_plot = R * torch.sin(Theta)
    
    # Generate full basis functions for plotting (only once)
    full_basis_functions = generate_basis_functions_for_plot(R, Theta, n_basis, bessel_zeros).detach().numpy()
    
    # Setup plot elements
    surface = None
    points = None
    title = ax.set_title("")
    
    # Set axis labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    # Set consistent axis limits
    max_z = 0
    min_z = 0
    
    # Find the global min and max Z values for consistent scaling
    for idx in range(len(X_tensor)):
        sample_X = X_tensor[idx]
        sample_y = y_tensor[idx]
        
        # Use the model to predict the coefficients
        with torch.no_grad():
            predicted_coefficients = model(sample_X.unsqueeze(0)).numpy().flatten()
        
        # Calculate the reconstructed shape
        reconstructed_shape = np.dot(full_basis_functions, predicted_coefficients)
        
        current_max = np.max(gap * reconstructed_shape)
        current_min = np.min(gap * reconstructed_shape)
        
        if current_max > max_z:
            max_z = current_max
        if current_min < min_z:
            min_z = current_min
    
    # Add a bit of padding
    max_z += 5
    min_z -= 5
    
    def update(frame):
        nonlocal surface, points
        ax.clear()
        
        sample_X = X_tensor[frame]
        sample_y = y_tensor[frame]
        sample_q = sample_X[0].item()  # Get voltage value
        
        # Extract coordinates from sample_X
        R_plot_sparse = sample_X[1::3]
        Theta_plot_sparse = sample_X[2::3]
        Z_plot_sparse = sample_X[3::3]
        
        X_plot_sparse = R_plot_sparse * torch.cos(Theta_plot_sparse)
        Y_plot_sparse = R_plot_sparse * torch.sin(Theta_plot_sparse)
        
        # Use the model to predict the coefficients
        with torch.no_grad():
            predicted_coefficients = model(sample_X.unsqueeze(0)).numpy().flatten()
        
        # Calculate the reconstructed shape
        reconstructed_shape = np.dot(full_basis_functions, predicted_coefficients)
        
        # Plotting the data points
        points = ax.scatter(radius*X_plot_sparse.numpy(), radius*Y_plot_sparse.numpy(), 
                           gap*Z_plot_sparse.numpy(), color='r', s=50, label='Data points')
        
        # Plot NN output surface
        surface = ax.plot_surface(radius*X_plot.numpy(), radius*Y_plot.numpy(), 
                                 gap*reconstructed_shape, alpha=0.5, cmap='plasma')
        
        # Setup consistent view
        ax.view_init(elev=30, azim=frame % 360, roll=0)  # Rotate view for better visualization
        ax.set_zlim(min_z, max_z)
        ax.set_title(f'Neural Network Prediction at Voltage = {sample_q:.2f}V')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        return [surface, points]
    
    frames = min(len(X_tensor), 50)  # Limit to 50 frames to avoid excessive rendering time
    ani = FuncAnimation(fig, update, frames=frames, interval=200, blit=False)
    
    # Save animation
    writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='ShapeNet'), bitrate=1800)
    output_path = 'animations/shape_prediction_matplotlib.mp4'
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved matplotlib animation to '{output_path}'")
    
    return output_path

def create_plotly_animation(model, X_tensor, y_tensor, n_basis, bessel_zeros, radius, gap):
    """
    Creates a plotly-based interactive animation of the model predictions.
    
    Args:
        model: ShapeNet model
        X_tensor: Input data tensor
        y_tensor: Target data tensor
        n_basis: Number of basis functions
        bessel_zeros: List of Bessel function zeros
        radius: Physical radius for scaling
        gap: Physical gap for scaling
        
    Returns:
        Path to the saved animation file
    """
    import os
    import plotly.graph_objects as go
    import plotly.io as pio
    
    # Create directory for animations if it doesn't exist
    os.makedirs("animations", exist_ok=True)
    
    # Generate mesh for plotting
    r_full = torch.linspace(0, 1, 100)
    theta_full = torch.linspace(0, 2 * np.pi, 100)
    Theta, R = torch.meshgrid(theta_full, r_full, indexing='ij')
    X_plot = radius * R * torch.cos(Theta)
    Y_plot = radius * R * torch.sin(Theta)
    
    # Generate full basis functions for plotting
    full_basis_functions = generate_basis_functions_for_plot(R, Theta, n_basis, bessel_zeros).detach().numpy()
    
    # Create frames for animation
    frames = []
    
    # Find global min and max for consistent color scale
    z_values = []
    for idx in range(len(X_tensor)):
        sample_X = X_tensor[idx]
        # Use the model to predict the coefficients
        with torch.no_grad():
            predicted_coefficients = model(sample_X.unsqueeze(0)).numpy().flatten()
        # Calculate the reconstructed shape
        reconstructed_shape = gap * np.dot(full_basis_functions, predicted_coefficients)
        z_values.extend(reconstructed_shape.flatten())
    
    z_min, z_max = min(z_values), max(z_values)
    
    # Create frames
    for idx in range(min(len(X_tensor), 100)):  # Limit to 100 frames
        sample_X = X_tensor[idx]
        sample_q = sample_X[0].item()  # Get voltage value
        
        # Extract coordinates from sample_X
        R_plot_sparse = sample_X[1::3]
        Theta_plot_sparse = sample_X[2::3]
        Z_plot_sparse = sample_X[3::3]
        
        X_plot_sparse = radius * R_plot_sparse * torch.cos(Theta_plot_sparse)
        Y_plot_sparse = radius * R_plot_sparse * torch.sin(Theta_plot_sparse)
        
        # Use the model to predict the coefficients
        with torch.no_grad():
            predicted_coefficients = model(sample_X.unsqueeze(0)).numpy().flatten()
        
        # Calculate the reconstructed shape
        reconstructed_shape = gap * np.dot(full_basis_functions, predicted_coefficients)
        
        # Create frame
        frame = go.Frame(
            data=[
                go.Surface(
                    x=X_plot.numpy(), 
                    y=Y_plot.numpy(), 
                    z=reconstructed_shape,
                    colorscale='Viridis',
                    opacity=0.7,
                    cmin=z_min,
                    cmax=z_max,
                    showscale=idx == 0,  # Only show colorbar in first frame
                ),
                go.Scatter3d(
                    x=X_plot_sparse.numpy(), 
                    y=Y_plot_sparse.numpy(), 
                    z=gap*Z_plot_sparse.numpy(),
                    mode='markers',
                    marker=dict(size=5, color='red'),
                )
            ],
            name=f"Voltage: {sample_q:.2f}V"
        )
        frames.append(frame)
    
    # Create initial surface for the first figure
    sample_X = X_tensor[0]
    sample_q = sample_X[0].item()
    
    # Extract coordinates from first sample
    R_plot_sparse = sample_X[1::3]
    Theta_plot_sparse = sample_X[2::3]
    Z_plot_sparse = sample_X[3::3]
    
    X_plot_sparse = radius * R_plot_sparse * torch.cos(Theta_plot_sparse)
    Y_plot_sparse = radius * R_plot_sparse * torch.sin(Theta_plot_sparse)
    
    # Use the model to predict the coefficients for first sample
    with torch.no_grad():
        predicted_coefficients = model(sample_X.unsqueeze(0)).numpy().flatten()
    
    # Calculate the reconstructed shape for first sample
    reconstructed_shape = gap * np.dot(full_basis_functions, predicted_coefficients)
    
    # Create the figure
    fig = go.Figure(
        data=[
            go.Surface(
                x=X_plot.numpy(), 
                y=Y_plot.numpy(), 
                z=reconstructed_shape,
                colorscale='Viridis',
                opacity=0.7,
                cmin=z_min,
                cmax=z_max
            ),
            go.Scatter3d(
                x=X_plot_sparse.numpy(), 
                y=Y_plot_sparse.numpy(), 
                z=gap*Z_plot_sparse.numpy(),
                mode='markers',
                marker=dict(size=5, color='red'),
                name='Data Points'
            )
        ],
        frames=frames
    )
    
    # Add animation controls
    fig.update_layout(
        title="ShapeNet Neural Network Prediction Animation",
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            zaxis=dict(range=[-30, 0]),  # Set z limits from -30 to 0
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
    output_path = "animations/shape_prediction_interactive.html"
    pio.write_html(fig, file=output_path, auto_open=False)
    print(f"Saved interactive Plotly animation to '{output_path}'")
    
    return output_path 