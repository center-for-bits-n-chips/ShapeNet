import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

def read_labview_binary(filename, num_markers=8, num_voltages=1, decimate=120):
    """
    Reads a binary file containing voltage and position data.
    
    Args:
        filename: Path to the binary file
        num_markers: Number of markers in the data (default: 8)
        num_voltages: Number of voltage measurements per record (default: 1)
        decimate: Take every Nth sample (default: 120)
    
    Returns:
        voltage: 1D array of voltage values
        positions: 3D array of shape (num_records, num_markers, 3) for x,y,z positions
    """
    # Calculate total values per record
    values_per_record = num_voltages + (num_markers * 3)  # voltages + (markers * xyz)
    
    # Read the entire file as double-precision floats
    data = np.fromfile(filename, dtype='<f8')
    
    # Calculate number of complete records
    num_records = data.size // values_per_record
    print(f"Number of records: {num_records}")

    # Discard data that's not fully written
    largest_multiple = (data.size // values_per_record) * values_per_record
    data = data[:largest_multiple]

    # Reshape so that each row corresponds to one complete record
    data = data.reshape((num_records, values_per_record))
    
    # Decimate the data
    data = data[::decimate]

    # Split into voltage and position data
    voltage = data[:, :num_voltages].squeeze()  # squeeze in case num_voltages=1
    positions_flat = data[:, num_voltages:]  # Shape is (num_records, 24)
    
    # Reorganize position data from [x1,x2,...,y1,y2,...,z1,z2,...] to (num_records, num_markers, xyz)
    x_coords = positions_flat[:, :num_markers]  # First 8 values are x coordinates
    y_coords = positions_flat[:, num_markers:2*num_markers]  # Next 8 are y coordinates
    z_coords = positions_flat[:, 2*num_markers:]  # Last 8 are z coordinates
    
    # Stack the coordinates to create (num_records, num_markers, 3) array
    positions = np.stack([x_coords, y_coords, z_coords], axis=2)
    
    # Calculate timing information
    original_sample_rate = 120  # Hz
    total_time = num_records / original_sample_rate  # seconds
    decimated_rate = original_sample_rate / decimate  # Hz
    
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Original sample rate: {original_sample_rate} Hz")
    print(f"Decimated sample rate: {decimated_rate:.2f} Hz")
    print(f"Number of decimated samples: {len(data)}")
    
    return voltage, positions

def create_circle_points(radius=250.0, num_points=100):
    """Create points for a circle in the XY plane centered at origin."""
    theta = np.linspace(0, 2*np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(theta)
    return np.column_stack((x, y, z))

def animate_markers(positions, z_scale=5.0, interval=50, save_path='marker_animation.mp4'):
    """
    Create a 3D animation of marker positions over time and save to file.
    
    Args:
        positions: Array of marker positions (time, markers, xyz)
        z_scale: Scale factor for z-displacement visualization (default: 5.0)
        interval: Animation interval in milliseconds (default: 50)
        save_path: Path to save the video file (default: 'marker_animation.mp4')
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scale z-coordinates for visualization
    scaled_positions = positions.copy()
    scaled_positions[:, :, 2] *= z_scale
    
    # Initialize scatter plot for markers
    scatter = ax.scatter(scaled_positions[0, :, 0], 
                        scaled_positions[0, :, 1], 
                        scaled_positions[0, :, 2])
    
    # Create and plot circle
    circle_points = create_circle_points()
    circle_line, = ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 
                          'g-', label='Reference Circle')
    
    # Set axis labels
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel(f'Z [mm] (scaled {z_scale}x)')
    
    # Find min and max values for consistent scaling
    x_min, x_max = min(positions[:, :, 0].min(), -250), max(positions[:, :, 0].max(), 250)
    y_min, y_max = min(positions[:, :, 1].min(), -250), max(positions[:, :, 1].max(), 250)
    z_min, z_max = scaled_positions[:, :, 2].min(), scaled_positions[:, :, 2].max()
    
    # Set axis limits with some padding
    padding = 10  # mm
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_zlim(z_min - padding, z_max + padding)
    
    # Add legend
    ax.legend()
    
    def update(frame):
        # Update scatter plot data with scaled z-coordinates
        scatter._offsets3d = (scaled_positions[frame, :, 0],
                            scaled_positions[frame, :, 1],
                            scaled_positions[frame, :, 2])
        ax.set_title(f'Frame {frame}')
        return scatter,
    
    # Create animation
    anim = animation.FuncAnimation(fig, update,
                                 frames=len(positions),
                                 interval=interval,
                                 blit=False)  # Changed to False for better compatibility
    
    # Set up the writer
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    
    # Save the animation
    anim.save(save_path, writer=writer)
    plt.close()
    
    print(f"Animation saved to {save_path}")

def main():
    print("\nReading data file...")
    filename = "pull-in.dat"
    voltage, positions = read_labview_binary(filename, decimate=120)
    print("\nGenerating plots...")
    
    # Create time series plot
    plt.figure(figsize=(10, 6))
    time_indices = np.arange(len(voltage))
    
    # Plot voltage
    plt.plot(time_indices, voltage, label='Voltage', color='black')
    
    # Plot each marker's z position
    colors = plt.cm.rainbow(np.linspace(0, 1, 8))  # Different color for each marker
    for i in range(8):
        plt.plot(time_indices, positions[:, i, 2],
                label=f'Marker {i} z',
                color=colors[i])

    plt.title('Motion Capture and Voltage Data vs Time')
    plt.xlabel('Sample Index')
    plt.ylabel('[mm], [kV]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Create voltage vs position plot
    plt.figure(figsize=(10, 6))
    for i in range(8):
        plt.plot(voltage, positions[:, i, 2],
                label=f'Marker {i} z',
                color=colors[i])
    
    plt.title('Z Position vs Voltage')
    plt.xlabel('Voltage [kV]')
    plt.ylabel('Z Position [mm]')
    plt.xlim(0, max(voltage) * 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Create and save 3D animation with scaled z-displacement
    #animate_markers(positions, z_scale=5.0, save_path='marker_animation.mp4')
    
    # Show the other plots
    plt.show()

if __name__ == "__main__":
    main()
