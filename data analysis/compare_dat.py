import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from read_dat import read_labview_binary, create_circle_points
from read_dic_data import get_displacement_data

def create_combined_animation(dic_data_frames, dat_file, output_file='combined_animation.mp4', z_scale=5.0, interval=50):
    """
    Create an animation combining DIC data and marker positions.
    
    Args:
        dic_data_frames: List of DIC data frames
        dat_file: Path to the .dat file
        output_file: Path to save the animation
        z_scale: Scale factor for z-displacement visualization
        interval: Animation interval in milliseconds
    """
    # Read the .dat file data
    voltage, positions, time = read_labview_binary(dat_file)
    
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    
    # Create 3D subplot for visualization
    ax_3d = fig.add_subplot(gs[0], projection='3d')
    
    # Create 2D subplot for time series
    ax_time = fig.add_subplot(gs[1])
    
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
                               -positions[0, :, 2],
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
    max_dic_displacement = [df['z-displacement[mm]'].min() for df in dic_data_frames]
    
    # Plot max DIC displacement
    time_line, = ax_time.plot(time[:len(max_dic_displacement)], max_dic_displacement, 
                            'b-', label='Max DIC Displacement')
    
    # Plot individual marker positions
    marker_lines = []
    colors = plt.cm.rainbow(np.linspace(0, 1, positions.shape[1]))
    for i in range(positions.shape[1]):
        line, = ax_time.plot(time, -positions[:, i, 2], 
                           color=colors[i], 
                           label=f'Marker {i}')
        marker_lines.append(line)
    
    # Add vertical line for current frame
    current_frame_line = ax_time.axvline(x=time[0], color='k', linestyle='--', alpha=0.5)
    
    # Set labels and title for time series plot
    ax_time.set_xlabel('Time [s]')
    ax_time.set_ylabel('Z Position [mm]')
    ax_time.set_title('Displacement vs Time')
    ax_time.grid(True)
    ax_time.legend()
    
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
    dic_directory = '18 mm test with sync'
    
    # Path to the .dat file
    dat_file = 'data/2025-03-31 pull-in DIC.dat'
    
    # Read DIC data
    print("Reading DIC data files...")
    dic_data_frames = get_displacement_data(dic_directory)
    print(f"Successfully read {len(dic_data_frames)} DIC frames")
    
    # Create combined animation
    print("Creating combined animation...")
    create_combined_animation(dic_data_frames, dat_file)

if __name__ == "__main__":
    main()
