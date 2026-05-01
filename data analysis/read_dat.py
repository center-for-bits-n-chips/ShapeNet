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
    plt.polar(theta[frame_index], r[frame_index], 'o', color='black')
    
    # Add text labels for each marker
    # for i in range(len(r[frame_index])):
    #     plt.text(theta[frame_index][i], r[frame_index][i], f'M{i}', 
    #             ha='center', va='bottom')
    
    plt.title(f'Marker Locations at t = {time[frame_index]:.2f} s')
    plt.ylim(0, 250)
    plt.grid(True)
    plt.rgrids(np.arange(0, 251, 75))  # Set radial ticks every 25 units from 0 to 250

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

def plot_multiple_profiles(positions, time, frame_indices, z_scale=1.0, fig=None, ax=None, label_prefix='', color='black'):
    """
    Plot multiple profiles on a figure, allowing multiple calls to overlay plots.
    
    Args:
        positions: 3D array of shape (num_records, num_markers, 3)
        time: Time array
        frame_indices: List of frame indices to plot
        z_scale: Scale factor for z-displacement visualization
        fig: Optional existing figure to plot on
        ax: Optional existing axes to plot on
        label_prefix: Optional prefix for legend labels
    """
    r, theta, z = cartesian_to_cylindrical(positions)
    z_scaled = z * z_scale
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    
    # Plot each profile
    for i, frame_idx in enumerate(frame_indices):
        # Sort r and z by increasing r values
        sort_idx = np.argsort(r[frame_idx])
        r_sorted = r[frame_idx][sort_idx]
        z_sorted = z_scaled[frame_idx][sort_idx]
        
        # Remove any duplicate r values by averaging corresponding z values
        unique_r, unique_idx = np.unique(r_sorted, return_index=True)
        unique_z = np.array([np.mean(z_sorted[r_sorted == r_val]) for r_val in unique_r])
        
        #label = f'{label_prefix}t = {time[frame_idx]:.2f} s' if label_prefix else f't = {time[frame_idx]:.2f} s'
        label = label_prefix
        ax.plot(unique_r, unique_z, 'o', label=label, color=color)
    
    ax.set_title('Profile Comparison')
    ax.set_xlabel('Radial Distance [mm]')
    ax.set_ylabel(f'Mesh Displacement [mm]')
    ax.legend()
    ax.grid(True)
    plt.xlim(0, 250)
    plt.ylim(0, 35)
    
    return fig, ax

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
    print("\nReading data files...")
    num_markers = 14
    num_voltages = 11
    
    # Read first file
    filename1 = "Lincoln Labs data/2025-06-06 DIC measurement final case B.dat"
    voltages1, positions1, time1 = read_labview_binary(filename1, num_markers=num_markers, num_voltages=num_voltages, decimate=1)
    
    # Read second file
    filename2 = "Lincoln Labs data/2025-06-06 DIC measurement final case A.dat"
    voltages2, positions2, time2 = read_labview_binary(filename2, num_markers=num_markers, num_voltages=num_voltages, decimate=1)
    
    print("\nGenerating plots...")

    # Plot marker locations at different frames
    plot_marker_locations(positions1, time1)
    plt.show()
            
    # Create time series plot
    plt.figure(figsize=(10, 6))
    
    # Plot voltages for first file
    colors_voltage = plt.cm.Greys(np.linspace(1.0, 0.2, num_voltages))  # Different greys for each voltage, darkest first
    for i in range(num_voltages):
        plt.plot(time1, voltages1[:, i],
                label=f'File 1 - Voltage {i}',
                color=colors_voltage[i],
                marker='.')
    
    # Plot each marker's z position for first file
    colors = plt.cm.rainbow(np.linspace(0, 1, num_markers))  # Different color for each marker
    for i in range(num_markers):
        plt.plot(time1, positions1[:, i, 2],
                label=f'File 1 - Marker {i} z',
                color=colors[i], 
                marker='.')
    
    # Plot voltages for second file with dashed lines
    for i in range(num_voltages):
        plt.plot(time2, voltages2[:, i],
                label=f'File 2 - Voltage {i}',
                color=colors_voltage[i],
                linestyle='--',
                marker='.')
    
    # Plot each marker's z position for second file with dashed lines
    for i in range(num_markers):
        plt.plot(time2, positions2[:, i, 2],
                label=f'File 2 - Marker {i} z',
                color=colors[i],
                linestyle='--',
                marker='.')

    plt.title('Motion Capture and Voltage Data vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('[mm], [kV]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)

    plt.show()
    
    # Create voltage vs position plot
    start_index_1 = 0
    end_index_1 = 38550
    start_index_2 = 16500
    end_index_2 = 53200
    plt.figure(figsize=(10, 6))
    
    # Plot data from first file
    mesh_voltage1 = voltages1[start_index_1:end_index_1, 0] - voltages1[start_index_1:end_index_1, 1]
    for i in [7]:
        plt.plot(mesh_voltage1, positions1[start_index_1:end_index_1, i, 2],
                #label=f'File 1 - Marker {i} z',
                label='Corrected',
                color='green')
    
    # Plot data from second file
    mesh_voltage2 = voltages2[start_index_2:end_index_2, 0] - voltages2[start_index_2:end_index_2, 1]
    for i in [7]:
        plt.plot(mesh_voltage2, positions2[start_index_2:end_index_2, i, 2],
                #label=f'File 2 - Marker {i} z',
                label='Uncorrected',
                color='red')
    
    # Read and plot COMSOL pull-in curves
    comsol_filename = "comsol/pull in 50-cm diameter 35-mm gap 0.5 N_per_m case A and B.txt"
    comsol_data = np.loadtxt(comsol_filename, skiprows=8)
    
    # First pull-in curve (Case B)
    voltage_b = comsol_data[:51, 1]  # kV
    displacement_b = comsol_data[:51, 0]  # mm
    plt.plot(voltage_b, displacement_b, '--', color='green', label='COMSOL Corrected')
    
    # Second pull-in curve (Case A)
    voltage_a = comsol_data[51:, 1]  # kV
    displacement_a = comsol_data[51:, 0]  # mm
    plt.plot(voltage_a, displacement_a, '--', color='red', label='COMSOL Uncorrected')
    
    # Fit polynomial to experimental data from first file
    voltage_poly, coeffs = fit_voltage_polynomial(positions1[start_index_1:end_index_1, 7, 2], mesh_voltage1, degree=3)
    
    # Test the polynomial with some example positions
    test_positions = np.linspace(positions1[start_index_1:end_index_1, 7, 2].min(), 35, 100)
    predicted_voltages = voltage_poly(test_positions)
    
    plt.title('Pull-In Curve')
    plt.xlabel('Voltage [kV]')
    plt.ylabel('Mesh Displacement [mm]')
    plt.xlim(0, 20)
    plt.ylim(0, 35)
    plt.legend(bbox_to_anchor=(0.8, 1), loc='upper left')
    plt.grid(True)

    plt.show()
    
    # Print polynomial coefficients and example predictions
    print("\nPolynomial coefficients:")
    for i, coef in enumerate(coeffs):
        print(f"x^{i+1}: {coef:.6f}")
    
    print("\nExample voltage predictions from position:")
    test_positions = np.linspace(positions1[start_index_1:end_index_1, 4, 2].min(), positions1[start_index_1:end_index_1, 4, 2].max(), 8)
    for pos in test_positions:
        voltage = voltage_poly(pos)
        print(f"Position: {pos:.1f} mm -> Voltage: {voltage:.2f} kV")
    
    # Plot multiple profiles
    fig, ax = plot_multiple_profiles(positions1, time1, frame_indices=[end_index_1], z_scale=1.0, label_prefix='Corrected', color='green')
    fig, ax = plot_multiple_profiles(positions2, time2, frame_indices=[end_index_2], z_scale=1.0, fig=fig, ax=ax, label_prefix='Uncorrected', color='red')
    
    # Add parabola
    r = np.linspace(0, 1, 100)  # Match x-axis limits
    nadir_depth = 24  # Scale to match y-axis max of 35
    y = -nadir_depth * (r**2 - 1)  # Parabola equation, shifted up to match y max
    ax.plot(250*r, y, '-', color='black', linewidth=2, label='f/D = 1.3 (24 mm)')
    
    # Read and plot COMSOL profiles
    comsol_filename = "comsol/profile 50-cm diameter 35-mm gap 1.7 N_per_m case A and B 24-mm depth.txt"
    comsol_data = np.loadtxt(comsol_filename, skiprows=8)
    
    # First profile (Case B)
    r_b = comsol_data[:347, 0] * 1e3  # Convert to mm
    z_b = -comsol_data[:347, 1]  # Convert to mm
    ax.plot(r_b, z_b, '--', color='red', label='COMSOL Uncorrected')
    
    # Second profile (Case A)
    r_a = comsol_data[348:, 0] * 1e3  # Convert to mm
    z_a = -comsol_data[348:, 1]  # Convert to mm
    ax.plot(r_a, z_a, '--', color='green', label='COMSOL Corrected')
    
    plt.legend(bbox_to_anchor=(0.6, 1), loc='upper left')
    plt.show()

    # Create and save 3D animation with scaled z-displacement
    #animate_markers(positions, z_scale=5.0, save_path='marker_animation.mp4')
    
    # Create and save profile animation
    #animate_profile(positions, time, z_scale=1.0, save_path='profile_animation.mp4')


if __name__ == "__main__":
    main()
