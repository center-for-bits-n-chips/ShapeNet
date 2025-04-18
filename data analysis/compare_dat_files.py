import numpy as np
import matplotlib.pyplot as plt
from read_dat import read_labview_binary

def plot_voltage_vs_position_comparison(file1, file2, labels=None, decimate=1, marker_indices=None, time_range1=None, time_range2=None):
    """
    Create a comparison plot of voltage vs z-position for two different .dat files.
    
    Args:
        file1: Path to first .dat file
        file2: Path to second .dat file
        labels: List of two strings for legend labels (default: uses filenames)
        decimate: Decimation factor for data reading (default: 1)
        marker_indices: List of marker indices to plot (default: all markers)
                       e.g. [0,1,2] would only plot the first three markers
        time_range1: Tuple of (start_time, end_time) in seconds to filter the first file's data
        time_range2: Tuple of (start_time, end_time) in seconds to filter the second file's data
    """
    # Read both files
    print(f"\nReading first file: {file1}")
    voltage1, positions1, time1 = read_labview_binary(file1, decimate=decimate)
    
    print(f"\nReading second file: {file2}")
    voltage2, positions2, time2 = read_labview_binary(file2, decimate=decimate)
    
    # Apply time range for first file if specified
    if time_range1 is not None:
        start_time, end_time = time_range1
        mask1 = (time1 >= start_time) & (time1 <= end_time)
        time1 = time1[mask1]
        voltage1 = voltage1[mask1]
        positions1 = positions1[mask1]
    
    # Apply time range for second file if specified
    if time_range2 is not None:
        start_time, end_time = time_range2
        mask2 = (time2 >= start_time) & (time2 <= end_time)
        time2 = time2[mask2]
        voltage2 = voltage2[mask2]
        positions2 = positions2[mask2]
    
    # Set up default labels if none provided
    if labels is None:
        import os
        labels = [os.path.basename(file1), os.path.basename(file2)]
    
    # If no marker indices specified, use all markers
    if marker_indices is None:
        marker_indices = range(positions1.shape[1])
    
    # Create the comparison plot
    plt.figure(figsize=(12, 8))
    
    # Use different color schemes for each dataset
    num_markers = len(marker_indices)
    colors1 = plt.cm.Blues(np.linspace(0.5, 1, num_markers))  # Blues for first dataset
    colors2 = plt.cm.Reds(np.linspace(0.5, 1, num_markers))   # Reds for second dataset
    
    # Plot first dataset
    for idx, i in enumerate(marker_indices):
        if idx == 0:  # Only add the label once for each dataset
            plt.plot(voltage1, positions1[:, i, 2],
                    color=colors1[idx], marker='.', linestyle='-',
                    label=f'{labels[0]} (Marker {i})', alpha=0.7)
        else:
            plt.plot(voltage1, positions1[:, i, 2],
                    color=colors1[idx], marker='.', linestyle='-',
                    label=f'{labels[0]} (Marker {i})', alpha=0.7)
    
    # Plot second dataset
    for idx, i in enumerate(marker_indices):
        if idx == 0:  # Only add the label once for each dataset
            plt.plot(voltage2, positions2[:, i, 2],
                    color=colors2[idx], marker='.', linestyle='-',
                    label=f'{labels[1]} (Marker {i})', alpha=0.7)
        else:
            plt.plot(voltage2, positions2[:, i, 2],
                    color=colors2[idx], marker='.', linestyle='-',
                    label=f'{labels[1]} (Marker {i})', alpha=0.7)
    
    plt.title('Z Position vs Voltage Comparison')
    plt.xlabel('Voltage [kV]')
    plt.ylabel('Z Position [mm]')
    
    # Set x-axis limits from 0 to max voltage
    max_voltage = max(voltage1.max(), voltage2.max())
    plt.xlim(0, max_voltage * 1.1)
    
    # Add grid and legend
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot
    plt.savefig('voltage_vs_position_comparison.png', bbox_inches='tight', dpi=300)
    plt.show()

def main():
    # Example usage
    file1 = "data/2025-04-16 pull-in 30 mm.dat"
    file2 = "data/2025-04-16 pull-in 0.01 mm_per_s.dat"
    # You can provide custom labels for the legend
    labels = ["open loop", "closed loop"]
    
    # Example: only plot markers 0, 3, and 7
    marker_indices = [0, 3, 7]
    
    # Separate time ranges for each file
    time_range1 = (0, 790)  # Time range for first file
    time_range2 = (0, 442.1)  # Time range for second file
    
    plot_voltage_vs_position_comparison(
        file1, 
        file2, 
        labels=labels, 
        decimate=120,
        marker_indices=marker_indices,
        time_range1=time_range1,
        time_range2=time_range2
    )

if __name__ == "__main__":
    main() 