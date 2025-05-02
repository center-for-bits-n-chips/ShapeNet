import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from glob import glob
from matplotlib.animation import FuncAnimation

def read_dic_file(file_path):
    """Read a single DIC CSV file and return the data as a pandas DataFrame."""
    return pd.read_csv(file_path, sep=';')

def get_dic_data(directory, DIC_sample_rate_s=1):
    """Read all DIC files in the directory and return a list of DataFrames with associated time and voltage data."""
    # Get all CSV files in the directory
    files = sorted(glob(os.path.join(directory, 'd*.csv')))
    
    # Read voltage data
    voltage_file = os.path.join(directory, 'V0001.csv')
    if os.path.exists(voltage_file):
        voltage_df = pd.read_csv(voltage_file, sep=';')
        # Get the voltage from 'Voltage ADC 1 channel 1' column
        voltage_kV = 6 * voltage_df['ADC 1 channel 1 [V]'].values.astype(float)
        time_s = DIC_sample_rate_s * voltage_df['Index'].values.astype(float)
        #time_s = 0.001 * voltage_df['Time'].values.astype(float)

    # Read each file and associate with corresponding time/voltage data
    data_frames = []
    for file in files:
        df = read_dic_file(file)
        data_frames.append(df)

    return data_frames, time_s, voltage_kV

def plot_membrane_deformation(data_frames, frame_index):
    """Plot the membrane deformation for a specific frame."""
    df = data_frames[frame_index]
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the deformed membrane
    scatter = ax.scatter(df['x[mm]'] + df['x-displacement[mm]'],
                        df['y[mm]'] + df['y-displacement[mm]'],
                        df['z[mm]'] + df['z-displacement[mm]'],
                        c=df['z-displacement[mm]'],
                        cmap='viridis',
                        s=1)
    
    # Add colorbar
    plt.colorbar(scatter, label='Z-displacement (mm)')
    
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Membrane Deformation - Frame {frame_index}')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    return fig

def create_animation(data_frames, output_file='membrane_deformation.gif'):
    """Create an animation of the membrane deformation."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize the scatter plot
    df = data_frames[0]
    scatter = ax.scatter(df['x[mm]'] + df['x-displacement[mm]'],
                        df['y[mm]'] + df['y-displacement[mm]'],
                        df['z[mm]'] + df['z-displacement[mm]'],
                        c=df['z-displacement[mm]'],
                        cmap='viridis',
                        s=1)
    
    # Add colorbar
    plt.colorbar(scatter, label='Z-displacement (mm)')
    
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Set fixed z limits
    ax.set_zlim(-20, 5)
    
    def update(frame):
        """Update function for animation."""
        df = data_frames[frame]
        scatter._offsets3d = (df['x[mm]'] + df['x-displacement[mm]'],
                            df['y[mm]'] + df['y-displacement[mm]'],
                            df['z[mm]'] + df['z-displacement[mm]'])
        scatter.set_array(df['z-displacement[mm]'])
        ax.set_title(f'Membrane Deformation - Frame {frame}')
        return scatter,
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(data_frames),
                        interval=50, blit=True)
    
    # Save animation
    anim.save(output_file, writer='pillow')
    plt.close()

def main(time_range=None):
    # Directory containing the DIC data
    directory = '28 mm closed loop take 1'
    
    # Read all data frames
    print("Reading DIC data files...")
    data_frames, time_s, voltage_kV = get_dic_data(directory)
    print(f"Successfully read {len(data_frames)} frames")
    
    # Create a time series plot of maximum displacement (using absolute values)
    max_displacements = []
    for df in data_frames:
        max_disp = abs(df['z-displacement[mm]']).max()
        max_displacements.append(max_disp)

    # Apply time range if specified
    if time_range is not None:
        start_time, end_time = time_range
        mask = (time_s >= start_time) & (time_s <= end_time)
        time_s = time_s[mask]
        max_displacements = np.array(max_displacements)[mask]
        voltage_kV = voltage_kV[mask]
        data_frames = [df for i, df in enumerate(data_frames) if mask[i]]
        print(f"Plotting data from {start_time:.3f}s to {end_time:.3f}s")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # First subplot: Time series
    ax1_twin = ax1.twinx()
    
    # Plot displacement on left y-axis
    line1 = ax1.plot(time_s, max_displacements, 'b-', label='Displacement')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Maximum Absolute Z-displacement (mm)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot voltage on right y-axis
    line2 = ax1_twin.plot(time_s, voltage_kV, 'r-', label='Voltage')
    ax1_twin.set_ylabel('Voltage (V)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    ax1.set_title('Membrane Displacement and Voltage Over Time')
    ax1.set_ylim(0, 30)
    ax1.grid(True)

    # Second subplot: Voltage vs Displacement
    ax2.plot(voltage_kV, max_displacements, '.', markersize=5)
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Maximum Absolute Z-displacement (mm)')
    ax2.set_title('Voltage vs Maximum Displacement')
    ax2.set_ylim(0, 30)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set time window parameters (in seconds)
    start_time = 25.0  # Set to None for full range, or specify a value like 1.0
    end_time = 275.0    # Set to None for full range, or specify a value like 2.0
    
    time_range = None
    if start_time is not None and end_time is not None:
        time_range = (start_time, end_time)
    
    main() 