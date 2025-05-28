import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit

def read_labview_binary(filename, num_markers=15, num_voltages=11, decimate=120):
    """
    Reads a binary file containing interleaved voltage and position data.
    Format: [voltage1, marker_data, voltage2...voltageN]
    
    Args:
        filename: Path to the binary file
        num_markers: Number of markers in the data (default: 15)
        num_voltages: Number of voltage measurements per record (default: 11)
        decimate: Take every Nth sample (default: 120)
    
    Returns:
        voltage: 1D array of voltage values
        positions: 3D array of shape (num_records, num_markers, 3) for x,y,z positions
        time: 1D array of time values in seconds
    """
    # Calculate total values per record
    values_per_record = (num_markers * 3) + num_voltages  # first voltage + marker data + remaining voltages
    
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

    # Extract voltages
    voltages = data[:, :num_voltages]
    
    # Extract marker data
    marker_data = data[:, num_voltages:]
    
    # Reorganize position data from [x1,x2,...,y1,y2,...,z1,z2,...] to (num_records, num_markers, xyz)
    x_coords = marker_data[:, :num_markers]  # First 15 values are x coordinates
    y_coords = marker_data[:, num_markers:2*num_markers]  # Next 15 are y coordinates
    z_coords = marker_data[:, 2*num_markers:]  # Last 15 are z coordinates
    
    # Stack the coordinates to create (num_records, num_markers, 3) array
    positions = np.stack([x_coords, y_coords, z_coords], axis=2)
    
    # Calculate time array
    original_sample_rate = 100  # Hz
    time = np.arange(num_records) / original_sample_rate  # Time in seconds
    time = time[::decimate]  # Decimate time array to match data
    decimated_rate = original_sample_rate / decimate  # Hz
    
    print(f"Total time: {time[-1]:.2f} seconds ({time[-1]/60:.2f} minutes)")
    print(f"Original sample rate: {original_sample_rate} Hz")
    print(f"Decimated sample rate: {decimated_rate:.2f} Hz")
    print(f"Number of decimated samples: {len(data)}")
    
    return voltages, positions, time

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

def cartesian_to_cylindrical(positions):
    """
    Convert Cartesian (x,y,z) coordinates to cylindrical (r,theta,z) coordinates.
    
    Args:
        positions: 3D array of shape (num_records, num_markers, 3) containing x,y,z coordinates
    
    Returns:
        r: Radial distance from origin
        theta: Angle in radians
        z: Height (unchanged)
    """
    x = positions[:, :, 0]
    y = positions[:, :, 1]
    z = positions[:, :, 2]
    
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    return r, theta, z

def plot_marker_locations(positions, time, frame_index=None):
    """
    Plot the profile of the shape in cylindrical coordinates.
    
    Args:
        positions: 3D array of shape (num_records, num_markers, 3)
        time: Time array
        frame_index: Optional index to plot a specific frame
    """
    r, theta, z = cartesian_to_cylindrical(positions)
    
    if frame_index is None:
        frame_index = 0  # Default to first frame
    
    plt.figure(figsize=(10, 6))
    plt.polar(theta[frame_index], r[frame_index], 'o')
    
    # Add text labels for each marker
    for i in range(len(r[frame_index])):
        plt.text(theta[frame_index][i], r[frame_index][i], f'M{i}', 
                ha='center', va='bottom')
    
    plt.title(f'Marker Locations at t = {time[frame_index]:.2f} s')
    plt.ylim(0, 250)
    plt.grid(True)

def animate_profile(positions, time, z_scale=5.0, interval=50, save_path='profile_animation.mp4'):
    """
    Create an animation of the profile over time and save to file.
    
    Args:
        positions: Array of marker positions (time, markers, xyz)
        time: Time array
        z_scale: Scale factor for z-displacement visualization
        interval: Animation interval in milliseconds
        save_path: Path to save the video file
    """
    r, theta, z = cartesian_to_cylindrical(positions)
    z_scaled = z * z_scale
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Initialize line plot
    line, = ax.plot([], [], '.-')
    
    # Initialize text annotations
    text_annotations = []
    for i in range(positions.shape[1]):
        annotation = ax.text(0, 0, '', animated=True)
        text_annotations.append(annotation)
    
    # Set axis limits
    ax.set_xlim(0, 300)  # Adjust based on your data
    ax.set_ylim(z_scaled.min(), z_scaled.max())
    
    # Set labels
    ax.set_xlabel('Radial Distance [mm]')
    ax.set_ylabel(f'Z Position [mm] (scaled {z_scale}x)')
    ax.grid(True)
    
    def update(frame):
        # Sort r and z by increasing r values
        sort_idx = np.argsort(r[frame])
        r_sorted = r[frame][sort_idx]
        z_sorted = z_scaled[frame][sort_idx]
        
        # Remove any duplicate r values by averaging corresponding z values
        unique_r, unique_idx = np.unique(r_sorted, return_index=True)
        unique_z = np.array([np.mean(z_sorted[r_sorted == r_val]) for r_val in unique_r])
        
        # Update line data
        line.set_data(unique_r, unique_z)
        ax.set_title(f'Profile at t = {time[frame]:.2f} s')
        
        # Update text annotations
        for i, (r_val, z_val) in enumerate(zip(r[frame], z_scaled[frame])):
            text_annotations[i].set_position((r_val, z_val))
            text_annotations[i].set_text(f'{z_val/z_scale:.1f}')  # Display unscaled z value
        
        return [line] + text_annotations
    
    # Create animation
    anim = animation.FuncAnimation(fig, update,
                                 frames=len(positions),
                                 interval=interval,
                                 blit=True)
    
    # Set up the writer
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    
    # Save the animation
    anim.save(save_path, writer=writer)
    plt.close()
    
    print(f"Profile animation saved to {save_path}")

def plot_multiple_profiles(positions, time, frame_indices, z_scale=5.0):
    """
    Plot multiple profiles on the same figure.
    
    Args:
        positions: 3D array of shape (num_records, num_markers, 3)
        time: Time array
        frame_indices: List of frame indices to plot
        z_scale: Scale factor for z-displacement visualization
    """
    r, theta, z = cartesian_to_cylindrical(positions)
    z_scaled = z * z_scale
    
    plt.figure(figsize=(10, 6))
    
    # Plot each profile
    for i, frame_idx in enumerate(frame_indices):
        # Sort r and z by increasing r values
        sort_idx = np.argsort(r[frame_idx])
        r_sorted = r[frame_idx][sort_idx]
        z_sorted = z_scaled[frame_idx][sort_idx]
        
        # Remove any duplicate r values by averaging corresponding z values
        unique_r, unique_idx = np.unique(r_sorted, return_index=True)
        unique_z = np.array([np.mean(z_sorted[r_sorted == r_val]) for r_val in unique_r])
        
        plt.plot(unique_r, unique_z, '.-', label=f't = {time[frame_idx]:.2f} s')
    
    plt.title('Profile Comparison')
    plt.xlabel('Radial Distance [mm]')
    plt.ylabel(f'Z Position [mm] (scaled {z_scale}x)')
    plt.legend()
    plt.grid(True)

def read_comsol_data(filename, set_index=0):
    """
    Read COMSOL simulation data from a text file.
    
    Args:
        filename: Path to the COMSOL data file
        set_index: Index of the data set to read (0-4)
        
    Returns:
        normalized_gap: Array of normalized gap values
        voltage: Array of voltage values
    """
    # Skip the header lines (first 8 lines)
    data = np.loadtxt(filename, skiprows=8)
    # Each set has 81 points
    points_per_set = 90
    start_idx = set_index * points_per_set
    end_idx = (set_index + 1) * points_per_set
    normalized_gap = data[start_idx:end_idx, 0]
    voltage = data[start_idx:end_idx, 1]
    return normalized_gap, voltage

def poly_through_origin(x, *coeffs):
    """
    Polynomial function that goes through (0,0).
    y = a₁x + a₂x² + a₃x³ + a₄x⁴
    """
    y = np.zeros_like(x)
    for i, coef in enumerate(coeffs):
        y += coef * x**(i+1)  # Start from x^1
    return y

def fit_voltage_polynomial(displacement, voltage, degree=4):
    """
    Fit a polynomial to predict voltage from position, forcing it through (0,0).
    
    Args:
        comsol_gap: Array of normalized gap values from COMSOL
        comsol_voltage: Array of voltage values from COMSOL
        initial_gap: Initial gap in mm (default: 35)
        degree: Degree of polynomial to fit (default: 4)
        
    Returns:
        poly: Function that takes position and returns voltage
        coeffs: Array of polynomial coefficients
    """
    # Initial guess for coefficients (all ones)
    p0 = np.ones(degree)
    
    # Fit the polynomial
    coeffs, _ = curve_fit(poly_through_origin, displacement, voltage, p0=p0)
    
    # Create a function that uses these coefficients
    def poly(x):
        return poly_through_origin(x, *coeffs)
    
    return poly, coeffs

def main():
    print("\nReading data file...")
    num_markers = 15
    num_voltages = 11
    
    filename = "data/2025-05-26 15 mm ramp 0.5 mmps.dat"
    voltages, positions, time = read_labview_binary(filename, num_markers=num_markers, num_voltages=num_voltages, decimate=1)
    print("\nGenerating plots...")

    # Plot marker locations at different frames
    plot_marker_locations(positions, time)
    plt.show()
            
    # Create time series plot
    plt.figure(figsize=(10, 6))
    
    # Plot voltages
    colors_voltage = plt.cm.Greys(np.linspace(1.0, 0.2, num_voltages))  # Different greys for each voltage, darkest first
    for i in range(num_voltages):
        plt.plot(time, voltages[:, i],
                label=f'Voltage {i}',
                color=colors_voltage[i],
                marker='.')
    
    # Plot each marker's z position
    colors = plt.cm.rainbow(np.linspace(0, 1, num_markers))  # Different color for each marker
    for i in range(num_markers):
        plt.plot(time, positions[:, i, 2],
                label=f'Marker {i} z',
                color=colors[i], 
                marker='.')

    plt.title('Motion Capture and Voltage Data vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('[mm], [kV]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)

    plt.show()
    
    # Create voltage vs position plot
    end_index = 228882
    plt.figure(figsize=(10, 6))
    
    mesh_voltage = voltages[:end_index, 0] - voltages[:end_index, 1]

    # Plot experimental data
    for i in [4]:
        plt.plot(mesh_voltage, positions[:end_index, i, 2],
                label=f'Marker {i} z',
                color=colors[i], 
                marker='.')
    
    # Fit polynomial to experimental data
    voltage_poly, coeffs = fit_voltage_polynomial(positions[:end_index, 4, 2], mesh_voltage, degree=5)
    
    # Test the polynomial with some example positions
    test_positions = np.linspace(positions[:end_index, 4, 2].min(), positions[:end_index, 4, 2].max(), 100)
    predicted_voltages = voltage_poly(test_positions)
    
    # Plot the polynomial fit
    plt.plot(predicted_voltages, test_positions, 'r:', label='Polynomial Fit', linewidth=1)
    
    plt.title('Z Position vs Voltage')
    plt.xlabel('Voltage [kV]')
    plt.ylabel('Z Position [mm]')
    plt.xlim(0, max(voltages[:, 0]) * 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    plt.show()
    
    # Print polynomial coefficients and example predictions
    print("\nPolynomial coefficients:")
    for i, coef in enumerate(coeffs):
        print(f"x^{i+1}: {coef:.6f}")
    
    print("\nExample voltage predictions from position:")
    test_positions = np.linspace(positions[:end_index, 4, 2].min(), positions[:end_index, 4, 2].max(), 8)
    for pos in test_positions:
        voltage = voltage_poly(pos)
        print(f"Position: {pos:.1f} mm -> Voltage: {voltage:.2f} kV")
    
    # Plot multiple profiles
    # plot_multiple_profiles(positions, time, frame_indices=[300, 400, 500], z_scale=1.0)
    # plt.show()

    # Create and save 3D animation with scaled z-displacement
    #animate_markers(positions, z_scale=5.0, save_path='marker_animation.mp4')
    
    # Create and save profile animation
    #animate_profile(positions, time, z_scale=1.0, save_path='profile_animation.mp4')


if __name__ == "__main__":
    main()
