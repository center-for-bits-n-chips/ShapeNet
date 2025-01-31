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
    data = np.fromfile(filename, dtype=np.float64)
    
    # Each record has 16 doubles total (8 for the first array, 8 for the second)
    # Figure out how many records there are
    num_records = data.size // 16
    
    # Reshape so that each row corresponds to 16 doubles (one record)
    data = data.reshape((num_records, 16))
    
    # Split into two arrays: first 8 columns, last 8 columns
    array1_all = data[:, :8]
    array2_all = data[:, 8:]
    
    return array1_all, array2_all

def main():
    # Path to your binary data file
    filename = "pull-in.dat"
    
    # Read the data
    array1_all, array2_all = read_labview_binary(filename)
    
    # For demonstration, we can plot the eight elements of array1
    # vs. record index. Similarly, we could do the same for array2.
    
    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot each column of array1
    for i in range(array1_all.shape[1]):
        plt.plot(array1_all[:, i], label=f'Array1 col {i}')

    # Optionally, plot each column of array2 in a similar fashion
    # for i in range(array2_all.shape[1]):
    #     plt.plot(array2_all[:, i], label=f'Array2 col {i}')

    plt.title('LabVIEW Binary Data (pull-in.dat)')
    plt.xlabel('Record index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()
