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

def get_dic_data(directory):
    """Read all DIC files in the directory and return a list of DataFrames."""
    # Get all CSV files in the directory
    files = sorted(glob(os.path.join(directory, 'd*.csv')))
    
    # Read voltage data
    voltage_file = os.path.join(directory, 'V0001.csv')
    voltage_data = None
    if os.path.exists(voltage_file):
        voltage_df = pd.read_csv(voltage_file, sep=';', skiprows=1, dtype={'V': float})
        # Get the voltage from 'Voltage ADC 1 channel 1' column
        voltage_data = voltage_df['V'].values.astype(float)

    # Read each file
    data_frames = []
    for file in files:
        df = read_dic_file(file)
        data_frames.append(df)
    
    return data_frames, voltage_data

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

def main():
    # Directory containing the DIC data
    directory = '28 mm closed loop lights off take 2'
    
    # Read all data frames
    print("Reading DIC data files...")
    data_frames, voltage_data = get_dic_data(directory)
    print(f"Successfully read {len(data_frames)} frames")
    
    # Plot a few key frames
    # frames_to_plot = [0, len(data_frames)//2, -1]  # First, middle, and last frame
    
    # for frame_idx in frames_to_plot:
    #     print(f"Plotting frame {frame_idx}...")
    #     fig = plot_membrane_deformation(data_frames, frame_idx)
    #     plt.show()
    #     plt.close()
    
    # Create animation
    # print("Creating animation...")
    # create_animation(data_frames)
    # print("Animation saved as 'membrane_deformation.gif'")
    
    # Create a time series plot of maximum displacement (using absolute values)
    max_displacements = []
    for df in data_frames:
        max_disp = abs(df['z-displacement[mm]']).max()
        max_displacements.append(max_disp)
    
    plt.figure(figsize=(10, 6))
    plt.plot(max_displacements)
    plt.xlabel('Frame')
    plt.ylabel('Maximum Absolute Z-displacement (mm)')
    plt.title('Maximum Membrane Displacement Over Time')
    plt.grid(True)
 
    # Plot voltage data
    plt.figure(figsize=(10, 6))
    plt.plot(voltage_data)
    plt.xlabel('Frame')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage Over Time')
    plt.show()

if __name__ == "__main__":
    main() 