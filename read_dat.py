def read_labview_binary(filename, num_markers=8, num_voltages=1, decimate=120):
    """
    Reads a binary file containing voltage and position data.
    
    Args:
        filename: Path to the binary file
        num_markers: Number of markers in the data (default: 8)
        num_voltages: Number of voltage measurements per record (default: 1)
        decimate: Take every Nth sample (default: 120)
    
    Returns:
        voltage: 1D array of voltage values
        positions: 3D array of shape (num_records, num_markers, 3) for x,y,z positions
    """
    # Calculate total values per record
    values_per_record = num_voltages + (num_markers * 3)  # voltages + (markers * xyz)
    
    # Read the entire file as double-precision floats
    data = np.fromfile(filename, dtype='<f8')
    
    # Calculate number of complete records
    num_records = data.size // values_per_record
    print(f"Number of records: {num_records}")

    # Discard data that's not fully written
    largest_multiple = (data.size // values_per_record) * values_per_record
    data = data[:largest_multiple]

    # Reshape so that each row corresponds to one complete record
    data = data.reshape((num_records, values_per_record))
    
    # Decimate the data
    data = data[::decimate]

    # Split into voltage and position data
    voltage = data[:, :num_voltages].squeeze()  # squeeze in case num_voltages=1
    positions_flat = data[:, num_voltages:]  # Shape is (num_records, 24)
    
    # Reorganize position data from [x1,x2,...,y1,y2,...,z1,z2,...] to (num_records, num_markers, xyz)
    x_coords = positions_flat[:, :num_markers]  # First 8 values are x coordinates
    y_coords = positions_flat[:, num_markers:2*num_markers]  # Next 8 are y coordinates
    z_coords = positions_flat[:, 2*num_markers:]  # Last 8 are z coordinates
    
    # Stack the coordinates to create (num_records, num_markers, 3) array
    positions = np.stack([x_coords, y_coords, z_coords], axis=2)
    
    # Calculate timing information
    original_sample_rate = 120  # Hz
    total_time = num_records / original_sample_rate  # seconds
    decimated_rate = original_sample_rate / decimate  # Hz
    
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Original sample rate: {original_sample_rate} Hz")
    print(f"Decimated sample rate: {decimated_rate:.2f} Hz")
    print(f"Number of decimated samples: {len(data)}")
    
    return voltage, positions

def main():
    print("\nReading data file...")
    filename = "2025-03-31 pull-in full.dat"
    voltage, positions = read_labview_binary(filename, decimate=120)  # uses defaults: num_markers=8, num_voltages=1
    # ... rest of the function ... 