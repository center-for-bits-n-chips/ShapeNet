import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from read_dat import read_labview_binary, create_circle_points
from read_dic import get_dic_data
from scipy.signal import correlate, resample_poly

def plot_time_series(dic_directory, dat_file, dic_time_range=None, time_shift=0.0):
    """
    Create and show a time series plot of DIC and marker data.
    
    Args:
        dic_directory: Directory containing DIC data
        dat_file: Path to the .dat file
        dic_time_range: Tuple of (start_time, end_time) in seconds to filter DIC data
        time_shift: Time shift in seconds to apply to marker data
    """
    # Read DIC data
    dic_data_frames, time_s, voltage_kV = get_dic_data(dic_directory)

    # Apply time range if specified
    if dic_time_range is not None:
        start_time, end_time = dic_time_range
        mask = (time_s >= start_time) & (time_s <= end_time)
        time_s = time_s[mask]
        voltage_kV = voltage_kV[mask]
        dic_data_frames = [df for i, df in enumerate(dic_data_frames) if mask[i]]
        print(f"Plotting data from {start_time:.3f}s to {end_time:.3f}s")

    # Read the .dat file data
    voltage, positions, time = read_labview_binary(dat_file, decimate=120*5)
    
    # Apply time shift to marker data
    time = time + time_shift
    print(f"Applied time shift: {time_shift:.3f} seconds")
    
    # Calculate max DIC displacement for each frame
    max_dic_displacement = [abs(df['z-displacement[mm]']).max() for df in dic_data_frames]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax_voltage = ax.twinx()
    
    # Plot max DIC displacement
    time_line, = ax.plot(time_s, max_dic_displacement, 
                        'b-', label='Max DIC Displacement')
    
    # Plot individual marker positions
    marker_lines = []
    colors = plt.cm.rainbow(np.linspace(0, 1, positions.shape[1]))
    for i in range(positions.shape[1]):
        line, = ax.plot(time, positions[:, i, 2], 
                       color=colors[i], 
                       label=f'Marker {i}')
        marker_lines.append(line)
    
    # Plot DIC voltage
    dic_voltage_line, = ax_voltage.plot(time_s, voltage_kV, 
                                       'r.-', label='DIC Voltage')
    
    # Plot marker voltage
    marker_voltage_line, = ax_voltage.plot(time, voltage, 
                                          'g.-', label='Marker Voltage')
    
    # Set labels and title
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Z Position [mm]')
    ax_voltage.set_ylabel('Voltage [kV]')
    ax.set_title('Displacement and Voltage vs Time')
    ax.grid(True)
    
    # Combine all lines for legend
    all_lines = [time_line] + marker_lines + [dic_voltage_line, marker_voltage_line]
    ax.legend(all_lines, [l.get_label() for l in all_lines], 
              bbox_to_anchor=(1.2, 1), loc='upper left')
    
    ax.set_xlim(0, 300)
    
    plt.tight_layout()
    plt.show()

def create_combined_animation(dic_directory, dat_file, dic_time_range=None, time_shift=0.0, output_file='combined_animation.mp4', z_scale=5.0, interval=50):
    """
    Create an animation combining DIC data and marker positions.
    
    Args:
        dic_data_frames: List of DIC data frames
        dat_file: Path to the .dat file
        dic_time_range: Tuple of (start_time, end_time) in seconds to filter DIC data
        time_shift: Time shift in seconds to apply to marker data (default: 0.0)
        output_file: Path to save the animation
        z_scale: Scale factor for z-displacement visualization
        interval: Animation interval in milliseconds
    """
    # Read DIC data
    dic_data_frames, time_s, voltage_kV = get_dic_data(dic_directory)

    # Apply time range if specified
    if dic_time_range is not None:
        start_time, end_time = dic_time_range
        mask = (time_s >= start_time) & (time_s <= end_time)
        time_s = time_s[mask]
        voltage_kV = voltage_kV[mask]
        dic_data_frames = [df for i, df in enumerate(dic_data_frames) if mask[i]]
        print(f"Plotting data from {start_time:.3f}s to {end_time:.3f}s")

    # Read the .dat file data
    voltage, positions, time = read_labview_binary(dat_file, decimate=120*5)
    
    # Apply time shift to marker data
    time = time + time_shift
    print(f"Applied time shift: {time_shift:.3f} seconds")
    
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    
    # Create 3D subplot for visualization
    ax_3d = fig.add_subplot(gs[0], projection='3d')
    
    # Create 2D subplot for time series
    ax_time = fig.add_subplot(gs[1])
    
    # Create twin axis for voltage
    ax_time_voltage = ax_time.twinx()
    
    # Create and plot reference circle
    circle_points = create_circle_points()
    circle_line, = ax_3d.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 
                           'g-', label='Reference Circle')
    
    # Initialize scatter plot for DIC data
    df = dic_data_frames[0]
    dic_scatter = ax_3d.scatter(df['x[mm]'] + df['x-displacement[mm]'],
                            df['y[mm]'] + df['y-displacement[mm]'],
                            df['z[mm]'] + df['z-displacement[mm]'],
                            c=df['z-displacement[mm]'],
                            cmap='viridis',
                            s=1,
                            label='DIC Data')
    
    # Add colorbar for DIC data
    plt.colorbar(dic_scatter, label='Z-displacement (mm)')
    
    # Initialize scatter plot for markers
    marker_scatter = ax_3d.scatter(positions[0, :, 0],
                               positions[0, :, 1],
                               positions[0, :, 2],
                               c='red',
                               s=100,
                               label='Markers')
    
    # Set labels and title for 3D plot
    ax_3d.set_xlabel('X [mm]')
    ax_3d.set_ylabel('Y [mm]')
    ax_3d.set_zlabel(f'Z [mm] (scaled {z_scale}x)')
    
    # Find min and max values for consistent scaling
    x_min, x_max = min(positions[:, :, 0].min(), -250), max(positions[:, :, 0].max(), 250)
    y_min, y_max = min(positions[:, :, 1].min(), -250), max(positions[:, :, 1].max(), 250)
    
    # Set axis limits with some padding
    padding = 10  # mm
    ax_3d.set_xlim(x_min - padding, x_max + padding)
    ax_3d.set_ylim(y_min - padding, y_max + padding)
    ax_3d.set_zlim(-20, 5)
    
    # Add legend to 3D plot
    ax_3d.legend()
    
    # Initialize time series plot
    # Calculate max DIC displacement for each frame
    max_dic_displacement = [abs(df['z-displacement[mm]']).max() for df in dic_data_frames]
    
    # Plot max DIC displacement
    time_line, = ax_time.plot(time_s, max_dic_displacement, 
                            'b-', label='Max DIC Displacement')
    
    # Plot individual marker positions
    marker_lines = []
    colors = plt.cm.rainbow(np.linspace(0, 1, positions.shape[1]))
    for i in range(positions.shape[1]):
        line, = ax_time.plot(time, positions[:, i, 2], 
                           color=colors[i], 
                           label=f'Marker {i}')
        marker_lines.append(line)
    
    # Plot DIC voltage
    dic_voltage_line, = ax_time_voltage.plot(time_s, voltage_kV, 
                                           'r-', label='DIC Voltage')
    
    # Plot marker voltage
    marker_voltage_line, = ax_time_voltage.plot(time, voltage, 
                                              'g-', label='Marker Voltage')
    
    # Add vertical line for current frame
    current_frame_line = ax_time.axvline(x=time[0], color='k', linestyle='--', alpha=0.5)
    
    # Set labels and title for time series plot
    ax_time.set_xlabel('Time [s]')
    ax_time.set_ylabel('Z Position [mm]')
    ax_time_voltage.set_ylabel('Voltage [kV]')
    ax_time.set_title('Displacement and Voltage vs Time')
    ax_time.grid(True)
    ax_time.set_xlim(0, 300)
    
    # Combine all lines for legend
    all_lines = [time_line] + marker_lines + [dic_voltage_line, marker_voltage_line]
    ax_time.legend(all_lines, [l.get_label() for l in all_lines], 
                  bbox_to_anchor=(1.2, 1), loc='upper left')
    
    def update(frame):
        """Update function for animation."""
        # Update 3D visualization
        df = dic_data_frames[frame]
        dic_scatter._offsets3d = (df['x[mm]'] + df['x-displacement[mm]'],
                                df['y[mm]'] + df['y-displacement[mm]'],
                                df['z[mm]'] + df['z-displacement[mm]'])
        dic_scatter.set_array(df['z-displacement[mm]'])
        
        marker_scatter._offsets3d = (positions[frame, :, 0],
                                   positions[frame, :, 1],
                                   -positions[frame, :, 2])
        
        ax_3d.set_title(f'Frame {frame}')
        
        # Update time series plot
        current_frame_line.set_xdata([time[frame], time[frame]])
        
        return [dic_scatter, marker_scatter, current_frame_line]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update,
                                frames=min(len(dic_data_frames), len(positions)),
                                interval=interval,
                                blit=False)
    
    # Set up the writer
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    
    # Save the animation
    anim.save(output_file, writer=writer)
    plt.close()
    
    print(f"Combined animation saved to {output_file}")

def main():
    # Directory containing the DIC data
    dic_directory = '28 mm closed loop lights off take 2'
    
    # Path to the .dat file
    dat_file = 'data/2025-04-03 pull-in 27.5 mm ramp then discharge.dat'
    
    # Set time window parameters (in seconds)
    start_time = 0.0  # Set to None for full range, or specify a value like 1.0
    end_time = None    # Set to None for full range, or specify a value like 2.0
    
    time_range = None
    if start_time is not None and end_time is not None:
        time_range = (start_time, end_time)
    
    # Set manual time shift (in seconds)
    time_shift = -25.0  # Adjust this value to align the signals
    
    # Show time series plot first
    print("Creating time series plot...")
    plot_time_series(dic_directory, dat_file, dic_time_range=time_range, time_shift=time_shift)
    
    # Create combined animation
    print("Creating combined animation...")
    create_combined_animation(dic_directory, dat_file, dic_time_range=time_range, time_shift=time_shift)

if __name__ == "__main__":
    main()
