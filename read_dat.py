import numpy as np
import matplotlib.pyplot as plt

def read_labview_binary(filename):
    """
    Reads a binary file where each record consists of two arrays of 8 doubles.
    Returns:
        array1_all: 2D NumPy array of shape (num_records, 8)
        array2_all: 2D NumPy array of shape (num_records, 8)
    """
    # Read the entire file as double-precision floats
    data = np.fromfile(filename, dtype='<f8')
    
    # Each record has 16 doubles total (8 for the first array, 8 for the second)
    # Figure out how many records there are
    num_records = data.size // 9

    # Discard data that's not fully written
    largest_multiple = (data.size // 9) * 9
    data = data[:largest_multiple]

    # Reshape so that each row corresponds to 16 doubles (one record)
    data = data.reshape((num_records, 9))

    # Write to CSV
    np.savetxt("output.csv", data, delimiter=",", fmt="%.4f")
    
    # Split into two arrays: first 8 columns, last 8 columns
    #data_initial = 500
    #data_final = 15000
    #voltage = data[data_initial:data_final, 0]
    #mocap = data[data_initial:data_final, 1:]
    voltage = data[:, 0]
    mocap = data[:, 1:]
    
    return voltage, mocap

def main():
    # Path to your binary data file
    filename = "pull-in.dat"
    # filename = "data/2025-01-31 ramp pull-in.dat"
    # filename = "data/impulse_data.dat"
    
    # Read the data
    voltage, mocap = read_labview_binary(filename)
    
    # For demonstration, we can plot the eight elements of array1
    # vs. record index. Similarly, we could do the same for array2.
    
    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot each column of array1
    for i in range(mocap.shape[1]):
        plt.plot(mocap[:, i], label=f'Mocap {i}')
    plt.plot(voltage, label=f'Voltage')

    plt.title('LabVIEW Binary Data (pull-in.dat)')
    plt.xlabel('Record index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    

    plt.figure(figsize=(10,6))
    # Plot each column of array1
    for i in range(mocap.shape[1]):
        plt.plot(voltage, mocap[:, i], label=f'Mocap {i}')
    plt.title('LabVIEW Binary Data (pull-in.dat)')
    plt.xlabel('Voltage [kV]')
    plt.ylabel('Displacement [mm]')
    plt.xlim(0, 25)
    plt.ylim(0, 35)
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
