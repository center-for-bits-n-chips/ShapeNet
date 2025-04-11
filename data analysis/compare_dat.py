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
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create and plot reference circle
    circle_points = create_circle_points()
    circle_line, = ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 
                          'g-', label='Reference Circle')
    
    # Initialize scatter plot for DIC data
    df = dic_data_frames[0]
    dic_scatter = ax.scatter(df['x[mm]'] + df['x-displacement[mm]'],
                           df['y[mm]'] + df['y-displacement[mm]'],
                           df['z[mm]'] + df['z-displacement[mm]'],
                           c=df['z-displacement[mm]'],
                           cmap='viridis',
                           s=1,
                           label='DIC Data')
    
    # Add colorbar for DIC data
    plt.colorbar(dic_scatter, label='Z-displacement (mm)')
    
    # Initialize scatter plot for markers
    marker_scatter = ax.scatter(positions[0, :, 0],
                              positions[0, :, 1],
                              positions[0, :, 2],
                              c='red',
                              s=100,
                              label='Markers')
    
    # Set labels and title
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel(f'Z [mm] (scaled {z_scale}x)')
    
    # Find min and max values for consistent scaling
    x_min, x_max = min(positions[:, :, 0].min(), -250), max(positions[:, :, 0].max(), 250)
    y_min, y_max = min(positions[:, :, 1].min(), -250), max(positions[:, :, 1].max(), 250)
    z_min, z_max = positions[:, :, 2].min() * z_scale, positions[:, :, 2].max() * z_scale
    
    # Set axis limits with some padding
    padding = 10  # mm
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_zlim(-20, 5)
    
    # Add legend
    ax.legend()
    
    def update(frame):
        """Update function for animation."""
        # Update DIC data
        df = dic_data_frames[frame]
        dic_scatter._offsets3d = (df['x[mm]'] + df['x-displacement[mm]'],
                                df['y[mm]'] + df['y-displacement[mm]'],
                                df['z[mm]'] + df['z-displacement[mm]'])
        dic_scatter.set_array(df['z-displacement[mm]'])
        
        # Update marker positions
        marker_scatter._offsets3d = (positions[frame, :, 0],
                                   positions[frame, :, 1],
                                   -positions[frame, :, 2])
        
        ax.set_title(f'Frame {frame}')
        return dic_scatter, marker_scatter
    
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
