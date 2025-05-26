import numpy as np
import pandas as pd
import torch
from scipy.special import j0, j1, jn_zeros
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import os
from shapenet_utils import generate_basis_functions
from ShapeNet.ShapeNet import ShapeNet
from read_dic import get_dic_data
from read_dat import read_labview_binary, cartesian_to_cylindrical

def load_model(model_filename, input_size, hidden_size, output_size):
    """Load the neural network model."""
    model = ShapeNet(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_filename))
    model.eval()  # Set model to evaluation mode
    return model

def prepare_labview_data_for_nn(positions, voltage):
    """Prepare LabVIEW data for neural network input."""
    # Convert to cylindrical coordinates
    r, theta, z = cartesian_to_cylindrical(positions)
    
    # Normalize radius
    r = r / 250.0  # Using radius=250mm as in the original code
    
    # Create input tensor
    nn_input = [voltage]  # Start with voltage
    for i in range(len(r)):
        nn_input.extend([r[i], theta[i], z[i]])
    
    return torch.tensor(nn_input, dtype=torch.float32)

def visualize_comparison(x, y, z, predicted_z, voltage, frame_idx):
    """Create a 3D visualization comparing DIC data with NN prediction."""
    # Create directory for visualizations if it doesn't exist
    os.makedirs("dic_nn_comparisons", exist_ok=True)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add DIC data points
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=z,
            colorscale='Viridis',
            opacity=0.8
        ),
        name='DIC Data'
    ))
    
    # Add predicted surface
    # Create a mesh for the surface
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Create a regular grid for the surface
    r_grid = np.linspace(0, np.max(r), 50)
    theta_grid = np.linspace(0, 2*np.pi, 50)
    R, Theta = np.meshgrid(r_grid, theta_grid)
    
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    # Interpolate predicted Z values onto the grid
    from scipy.interpolate import griddata
    Z = griddata((x, y), predicted_z, (X, Y), method='cubic')
    
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        opacity=0.5,
        name='NN Prediction'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'DIC vs NN Prediction (Voltage: {voltage:.2f}V, Frame: {frame_idx})',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        )
    )
    
    # Save the figure
    filename = f"dic_nn_comparisons/comparison_frame_{frame_idx}.html"
    fig.write_html(filename)
    print(f"Saved comparison visualization to {filename}")
    
    return filename

def visualize_error_animation(x, y, z, predicted_z, voltage, frame_idx, error_data):
    """Create an animated 3D visualization of the error between DIC and NN prediction."""
    # Calculate error
    error = z - predicted_z
    
    # Store error data for animation
    error_data.append({
        'frame': frame_idx,
        'voltage': voltage,
        'rmse': np.sqrt(np.mean(error**2)),
        'max_error': np.max(np.abs(error)),
        'mean_error': np.mean(error),
        'x': x,
        'y': y,
        'error': error
    })
    
    return None

def create_error_animation(error_data):
    """Create a single animated plot showing error evolution over time."""
    # Create directory for visualizations if it doesn't exist
    os.makedirs("error_animations", exist_ok=True)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add initial frame
    first_frame = error_data[0]
    error = first_frame['error']
    x = first_frame['x']
    y = first_frame['y']
    
    # Add error points
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=error,
        mode='markers',
        marker=dict(
            size=2,
            color=error,
            colorscale='RdBu',  # Red-Blue colormap for positive/negative errors
            opacity=0.8,
            colorbar=dict(title='Error (mm)')
        ),
        name='Error'
    ))
    
    # Create frames for animation
    frames = []
    for frame_data in error_data:
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=frame_data['x'],
                    y=frame_data['y'],
                    z=frame_data['error'],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=frame_data['error'],
                        colorscale='RdBu',
                        opacity=0.8
                    )
                )
            ],
            name=f"frame_{frame_data['frame']}"
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Add animation buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=100, redraw=True),
                                fromcurrent=True,
                                mode="immediate"
                            )
                        ]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0)
                            )
                        ]
                    )
                ],
                direction="left",
                pad=dict(r=10, t=10),
                showactive=False,
                x=0.1,
                y=0,
                xanchor="right",
                yanchor="top"
            )
        ],
        sliders=[
            dict(
                currentvalue=dict(prefix="Frame: "),
                pad=dict(t=50),
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [f"frame_{frame_data['frame']}"],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0)
                            )
                        ],
                        label=f"{frame_data['frame']}"
                    )
                    for frame_data in error_data
                ]
            )
        ]
    )
    
    # Update layout
    fig.update_layout(
        title='Error Evolution Over Time',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Error (mm)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        showlegend=False
    )
    
    # Save the animation
    filename = "error_animations/error_animation.html"
    fig.write_html(filename)
    print(f"\nSaved error animation to {filename}")
    
    # Create error metrics plot
    error_df = pd.DataFrame([{
        'frame': d['frame'],
        'voltage': d['voltage'],
        'rmse': d['rmse'],
        'max_error': d['max_error'],
        'mean_error': d['mean_error']
    } for d in error_data])
    
    metrics_fig = go.Figure()
    
    metrics_fig.add_trace(go.Scatter(
        x=error_df['frame'],
        y=error_df['rmse'],
        mode='lines+markers',
        name='RMSE'
    ))
    
    metrics_fig.add_trace(go.Scatter(
        x=error_df['frame'],
        y=error_df['max_error'],
        mode='lines+markers',
        name='Max Error'
    ))
    
    metrics_fig.add_trace(go.Scatter(
        x=error_df['frame'],
        y=error_df['mean_error'],
        mode='lines+markers',
        name='Mean Error'
    ))
    
    metrics_fig.update_layout(
        title='Error Metrics Over Time',
        xaxis_title='Frame',
        yaxis_title='Error (mm)',
        hovermode='x unified'
    )
    
    # Save the error metrics plot
    metrics_fig.write_html("error_animations/error_metrics.html")
    print("Saved error metrics visualization to error_animations/error_metrics.html")

def main():
    # Parameters
    n_basis = 2  # Must match the training parameters
    n_markers = 8
    hidden_size = 32
    model_filename = 'shape_net_model.pth'
    time_shift = -7.5  # Time shift to align DIC and LabVIEW data
    
    # Initialize list to store error data for animation
    error_data = []
    
    # Load DIC data
    print("Loading DIC data...")
    directory = '28 mm closed loop take 1'
    dic_data_frames, dic_time_s, dic_voltage_kV = get_dic_data(directory)
    print(f"Loaded {len(dic_data_frames)} DIC frames")
    
    # Load LabVIEW data
    print("\nLoading LabVIEW data...")
    dat_file = 'data/2025-04-22 28 mm pull-in take 1.dat'
    labview_voltage, labview_positions, labview_time = read_labview_binary(dat_file, decimate=10)
    labview_time = labview_time + time_shift  # Apply time shift
    print(f"Loaded {len(labview_time)} LabVIEW frames")
    
    # Convert all LabVIEW positions to cylindrical coordinates
    print("\nConverting LabVIEW positions to cylindrical coordinates...")
    labview_r, labview_theta, labview_z = cartesian_to_cylindrical(labview_positions)
    
    # Load neural network model
    print("\nLoading neural network model...")
    input_size = 1 + 3 * n_markers  # voltage + (r, theta, z) for each marker
    output_size = 3 * n_basis  # coefficients for each basis function
    model = load_model(model_filename, input_size, hidden_size, output_size)
    
    # Generate Bessel zeros
    bessel_zeros = []
    for m in range(2):
        zeros = jn_zeros(m, n_basis)
        bessel_zeros.extend(zeros)
    
    # Process each frame
    print("\nProcessing frames...")
    for frame_idx, (dic_df, dic_time) in enumerate(zip(dic_data_frames, dic_time_s)):
        # Find closest LabVIEW frame
        labview_idx = np.argmin(np.abs(labview_time - dic_time))
        if abs(labview_time[labview_idx] - dic_time) > 0.1:  # Skip if time difference is too large
            print(f"Skipping frame {frame_idx} - no matching LabVIEW data")
            continue
            
        print(f"\nProcessing frame {frame_idx}")
        print(f"DIC time: {dic_time:.3f}s, LabVIEW time: {labview_time[labview_idx]:.3f}s")
        
        # Get DIC coordinates
        x = dic_df['x[mm]'] + dic_df['x-displacement[mm]']
        y = dic_df['y[mm]'] + dic_df['y-displacement[mm]']
        z = dic_df['z[mm]'] + dic_df['z-displacement[mm]']
        voltage = labview_voltage[labview_idx]  # Use LabVIEW voltage
        
        # Create input tensor from the current frame's cylindrical coordinates
        nn_input = [voltage]  # Start with voltage
        for i in range(n_markers):
            nn_input.extend([
                labview_r[labview_idx, i] / 250.0,  # Normalize radius
                labview_theta[labview_idx, i],
                labview_z[labview_idx, i]
            ])
        nn_input = torch.tensor(nn_input, dtype=torch.float32)
        
        # Get neural network prediction
        with torch.no_grad():
            predicted_coefficients = model(nn_input.unsqueeze(0)).numpy().flatten()
        
        # Generate basis functions for all DIC points
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        r_normalized = r / 250.0  # Normalize radius
        
        # Convert to torch tensors for basis function generation
        r_tensor = torch.tensor(r_normalized)
        theta_tensor = torch.tensor(theta)
        
        # Generate basis functions
        basis_functions = generate_basis_functions(r_tensor, theta_tensor, n_basis, bessel_zeros)
        
        # Calculate predicted shape
        predicted_z = -np.dot(basis_functions.detach().numpy(), predicted_coefficients)
        
        # Visualize comparison
        visualize_comparison(x, y, z, predicted_z, voltage, frame_idx)
        
        # Store error data for animation
        visualize_error_animation(x, y, z, predicted_z, voltage, frame_idx, error_data)
        
        # Calculate and print error metrics
        rmse = np.sqrt(np.mean((z - predicted_z)**2))
        print(f"Mean Squared Error: {rmse:.4f}")
    
    # Create the error animation
    create_error_animation(error_data)

if __name__ == "__main__":
    main() 