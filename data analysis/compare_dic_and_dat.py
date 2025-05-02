import numpy as np
import matplotlib.pyplot as plt
from read_dat import read_labview_binary
from read_dic import get_dic_data
from scipy.spatial.distance import cdist

def find_nearest_dic_points(data_frames, marker_positions, marker_indices):
    """
    Find the nearest DIC points to each marker based on x,y coordinates.
    
    Args:
        data_frames: List of DIC data frames
        marker_positions: Array of marker positions
        marker_indices: List of marker indices to consider
    
    Returns:
        Dictionary mapping marker indices to their nearest DIC point indices
    """
    # Get x,y coordinates from first DIC frame
    dic_points = data_frames[0][['x[mm]', 'y[mm]']].values
    
    # Dictionary to store results
    nearest_points = {}
    
    for i in marker_indices:
        # Get marker's x,y position (using first frame)
        marker_xy = marker_positions[0, i, :2]  # First frame, marker i, x and y coordinates
        
        # Calculate distances to all DIC points
        distances = cdist([marker_xy], dic_points)[0]
        
        # Find nearest point
        nearest_idx = np.argmin(distances)
        nearest_points[i] = nearest_idx
        
        print(f"Marker {i} (x={marker_xy[0]:.2f}, y={marker_xy[1]:.2f}) -> DIC Point {nearest_idx} (x={dic_points[nearest_idx, 0]:.2f}, y={dic_points[nearest_idx, 1]:.2f})")
    
    return nearest_points

def plot_combined_voltage_vs_displacement(dic_directory, dat_file, labels=None, decimate=1, marker_indices=None, dic_time_range=None, marker_time_range=None):
    """
    Create a combined plot of voltage vs displacement for both DIC and marker data.
    
    Args:
        dic_directory: Directory containing DIC data files
        dat_file: Path to the .dat file
        labels: List of two strings for legend labels (default: ['DIC', 'Markers'])
        decimate: Decimation factor for data reading (default: 1)
        marker_indices: List of marker indices to plot (default: all markers)
        dic_time_range: Tuple of (start_time, end_time) in seconds to filter DIC data
        marker_time_range: Tuple of (start_time, end_time) in seconds to filter marker data
    """
    # Read DIC data
    print("Reading DIC data files...")
    data_frames, time_s, voltage_kV = get_dic_data(dic_directory)
    print(f"Successfully read {len(data_frames)} DIC frames")
    
    # Read marker data
    print(f"\nReading marker data file: {dat_file}")
    voltage_markers, positions, time_markers = read_labview_binary(dat_file, decimate=decimate)
    
    # Apply time range for DIC data if specified
    if dic_time_range is not None:
        start_time, end_time = dic_time_range
        mask_dic = (time_s >= start_time) & (time_s <= end_time)
        time_s = time_s[mask_dic]
        voltage_kV = voltage_kV[mask_dic]
        data_frames = [df for i, df in enumerate(data_frames) if mask_dic[i]]
    
    # Apply time range for marker data if specified
    if marker_time_range is not None:
        start_time, end_time = marker_time_range
        mask_markers = (time_markers >= start_time) & (time_markers <= end_time)
        time_markers = time_markers[mask_markers]
        voltage_markers = voltage_markers[mask_markers]
        positions = positions[mask_markers]
    
    # Set up default labels if none provided
    if labels is None:
        labels = ['DIC', 'Markers']
    
    # If no marker indices specified, use all markers
    if marker_indices is None:
        marker_indices = range(positions.shape[1])
    
    # Find nearest DIC points to each marker
    print("\nFinding nearest DIC points to markers:")
    nearest_points = find_nearest_dic_points(data_frames, positions, marker_indices)
    
    # Create the comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot nearest DIC points in different colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(marker_indices)))
    for idx, i in enumerate(marker_indices):
        point_idx = nearest_points[i]
        point_displacements = np.array([df.iloc[point_idx]['z-displacement[mm]'] for df in data_frames])
        plt.plot(voltage_kV, -point_displacements, color=colors[idx], linewidth=2, marker='x',
                label=f'DIC Point {point_idx} (Nearest to Marker {i})')
    
    # Plot marker data
    colors = plt.cm.Reds(np.linspace(0.5, 1, len(marker_indices)))
    for idx, i in enumerate(marker_indices):
        if idx == 0:  # Only add the label once for markers
            plt.plot(voltage_markers, positions[:, i, 2],
                    color=colors[idx], marker='.', linestyle='-',
                    label=f'{labels[1]} (Marker {i})', alpha=0.7)
        else:
            plt.plot(voltage_markers, positions[:, i, 2],
                    color=colors[idx], marker='.', linestyle='-',
                    label=f'{labels[1]} (Marker {i})', alpha=0.7)
    
    plt.title('Voltage vs Displacement Comparison with Nearest DIC Points')
    plt.xlabel('Voltage [kV]')
    plt.ylabel('Displacement [mm]')
    
    # Set x-axis limits from 0 to max voltage
    max_voltage = max(voltage_kV.max(), voltage_markers.max())
    plt.xlim(0, max_voltage * 1.1)
    
    # Add grid and legend
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot
    plt.savefig('DIC_and_mocap_voltage_vs_displacement_comparison.png', bbox_inches='tight', dpi=300)
    plt.show()

def main():
    # Example usage
    dic_directory = '28 mm closed loop take 1'
    dat_file = "data/2025-04-22 28 mm pull-in take 1.dat"
    
    # Custom labels
    labels = ["DIC (Closed Loop)", "Markers (Closed Loop)"]
    
    # Example: only plot markers 0, 3, and 7
    marker_indices = [0, 3, 7]
    
    # Separate time ranges for DIC and marker data (in seconds)
    dic_time_range = (0.0, 275)  # Time range for DIC data
    marker_time_range = (0.0, 600)  # Different time range for marker data
    
    plot_combined_voltage_vs_displacement(
        dic_directory,
        dat_file,
        labels=labels,
        decimate=120,
        marker_indices=marker_indices,
        #dic_time_range=dic_time_range,
        marker_time_range=marker_time_range
    )

if __name__ == "__main__":
    main() 