import numpy as np
import matplotlib.pyplot as plt

def read_labview_binary(filename):
    """
    Reads a binary file where each record consists of voltage and mocap data.
    Returns:
        voltage: Array of voltage values
        mocap: 2D array of mocap positions
    """
    # Read the entire file as double-precision floats
    data = np.fromfile(filename, dtype='<f8')
    
    # Each record has 9 doubles (1 for voltage, 8 for mocap)
    num_records = data.size // 9
    largest_multiple = (data.size // 9) * 9
    data = data[:largest_multiple]
    data = data.reshape((num_records, 9))
    
    voltage = data[:, 0]
    mocap = data[:, 1:]
    
    return voltage, mocap

def calculate_spread(positions):
    """Calculate the spread (max - min) of positions"""
    return np.max(positions) - np.min(positions)

def plot_comparison():
    # Read both datasets
    voltage1, mocap1 = read_labview_binary("2025-03-31 pull-in rim 1.dat")
    voltage2, mocap2 = read_labview_binary("2025-03-31 pull-in plane 15 mm.dat")

    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Get initial and final positions
    initial_pos1 = mocap1[0, :]
    final_pos1 = mocap1[-1, :]
    initial_pos2 = mocap2[0, :]
    final_pos2 = mocap2[-1, :]
    
    # Calculate spreads
    spread_initial1 = calculate_spread(initial_pos1)
    spread_final1 = calculate_spread(final_pos1)
    spread_initial2 = calculate_spread(initial_pos2)
    spread_final2 = calculate_spread(final_pos2)
    
    # Create x-coordinates for the markers
    markers = np.arange(mocap1.shape[1])
    
    # Plot initial and final positions
    plt.plot(markers, initial_pos1, 'ro-', label='Rim 1 Initial', markersize=8)
    plt.plot(markers, final_pos1, 'ro--', label='Rim 1 Final', markersize=8)
    plt.plot(markers, initial_pos2, 'bs-', label='Plane Initial', markersize=8)
    plt.plot(markers, final_pos2, 'bs--', label='Plane Final', markersize=8)
    
    # Add displacement annotations
    for i in markers:
        displacement1 = final_pos1[i] - initial_pos1[i]
        displacement2 = final_pos2[i] - initial_pos2[i]
        plt.annotate(f'Δ={displacement1:.1f}', 
                    xy=(i, (final_pos1[i] + initial_pos1[i])/2),
                    xytext=(10, 0), textcoords='offset points')
        plt.annotate(f'Δ={displacement2:.1f}', 
                    xy=(i, (final_pos2[i] + initial_pos2[i])/2),
                    xytext=(10, 0), textcoords='offset points')

    # Add spread information to title
    plt.title('Initial vs Final Positions\n' + 
              f'Rim 1 spread: {spread_initial1:.1f}→{spread_final1:.1f} mm\n' +
              f'Plane spread: {spread_initial2:.1f}→{spread_final2:.1f} mm')
    plt.xlabel('Marker Number')
    plt.ylabel('Position [mm]')
    plt.grid(True)
    plt.legend()
    plt.xticks(markers)

    plt.tight_layout()
    plt.show()

    # Print detailed spread information
    print(f"\nSpread Analysis:")
    print(f"Rim 1:")
    print(f"  Initial spread: {spread_initial1:.1f} mm")
    print(f"  Final spread: {spread_final1:.1f} mm")
    print(f"  Change in spread: {spread_final1 - spread_initial1:.1f} mm")
    print(f"\nPlane:")
    print(f"  Initial spread: {spread_initial2:.1f} mm")
    print(f"  Final spread: {spread_final2:.1f} mm")
    print(f"  Change in spread: {spread_final2 - spread_initial2:.1f} mm")

if __name__ == "__main__":
    plot_comparison()