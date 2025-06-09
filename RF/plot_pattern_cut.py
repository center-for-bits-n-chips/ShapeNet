import numpy as np
import matplotlib.pyplot as plt

def find_hpbw(angles, gains):
    """Find the Half Power Beam Width (HPBW) of the main beam."""
    max_gain = np.max(gains)
    half_power = max_gain - 3  # 3 dB down from peak
    
    # Find points where gain crosses half power level
    crossings = []
    for i in range(len(gains)-1):
        if (gains[i] >= half_power and gains[i+1] < half_power) or \
           (gains[i] < half_power and gains[i+1] >= half_power):
            # Linear interpolation to find exact crossing point
            x1, x2 = angles[i], angles[i+1]
            y1, y2 = gains[i], gains[i+1]
            x_cross = x1 + (x2-x1)*(half_power-y1)/(y2-y1)
            crossings.append(x_cross)
    
    if len(crossings) >= 2:
        # Find the two crossings closest to the peak
        peak_idx = np.argmax(gains)
        peak_angle = angles[peak_idx]
        crossings = sorted(crossings, key=lambda x: abs(x - peak_angle))
        hpbw = abs(crossings[1] - crossings[0])
        return hpbw, crossings[0], crossings[1]
    return None, None, None

def find_sidelobes(angles, gains):
    """Find the first sidelobe levels on both sides of the main beam."""
    max_gain = np.max(gains)
    peak_idx = np.argmax(gains)
    peak_angle = angles[peak_idx]
    
    # Find local maxima in the gain pattern
    local_maxima = []
    for i in range(1, len(gains)-1):
        if gains[i] > gains[i-1] and gains[i] > gains[i+1]:
            local_maxima.append((angles[i], gains[i]))
    
    # Sort by distance from peak
    local_maxima.sort(key=lambda x: abs(x[0] - peak_angle))
    
    # Get the first two sidelobes (one on each side)
    sidelobes = []
    for angle, gain in local_maxima:
        if angle < peak_angle and len(sidelobes) < 1:
            sidelobes.append((angle, gain))
        elif angle > peak_angle and len(sidelobes) < 2:
            sidelobes.append((angle, gain))
        if len(sidelobes) == 2:
            break
    
    return sidelobes

# Parameter to specify frequency in GHz
FREQUENCY_GHZ = 35  # Options: 30, 35, or 40

wavelength = 299792458 / (FREQUENCY_GHZ * 1e9) # meters
diameter = 0.5 # meters

predicted_hpbw = 70 * wavelength / diameter
eta = 0.6
predicted_gain =  20 * np.log10(eta * np.pi * diameter / wavelength)
print(f"Predicted HPBW: {predicted_hpbw:.2f} degrees, Predicted gain: {predicted_gain:.2f} dBi")

# List of file paths to plot
file_paths = [
    #'LL measurements/EAR_PEAK_AZ_LONG_CUT_5_29_2025_002.std',
    #'LL measurements/EAR_PEAK_AZ_LONG_CUT_2_5_29_2025_003.std',
    #'LL measurements/EAR_PEAK_AZ_LONG_CUT_3_5_29_2025_007.std',
    #'LL measurements/EAR_PEAK_AZ_LONG_CUT_6_5_30_2025_019.std',
    #'LL measurements/EAR_PEAK_AZ_CUT_6_REPEAT_6_2_2025_028.std',
    #'LL measurements/EAR_PEAK_AZ_CUT_8_6_3_2025_030.std',
    #'LL measurements/EAR_DISH_AZ_CUT_9_6_3_2025_039.std',
    #'LL measurements/EAR_DISH_AZ_CUT_10_6_3_2025_040.std',
    #'LL measurements/EAR_DISH_AZ_CUT_11_6_3_2025_041.std',
    #'LL measurements/EAR_DISH_AZ_CUT_12_6_3_2025_042.std', # the best one
    #'LL measurements/EAR_DISH_EL_CUT_13_6_3_2025_044.std',
    #'LL measurements/EAR_DISH_EL_CUT_22_6_3_2025_055.std',
    'LL measurements/EAR_DISH_AZ_CUT_LONG_23_6_3_2025_056.std',
    'LL measurements/EAR_DISH_AZ_LONG_CUT_31_6_3_2025_064.std',
]

# Convert target frequency from GHz to MHz
target_freq_mhz = FREQUENCY_GHZ * 1000

# Create the plot
plt.figure(figsize=(8, 8))  # Made taller to accommodate both plots
plt.subplot(2, 1, 1)  # Top subplot for gain

# Read FEKO simulation data
feko_data = np.loadtxt('measured pattern 30 GHz COMSOL f_D 1.3', skiprows=2)
feko_angles = feko_data[:, 0]
feko_gains = feko_data[:, 1]

# Process each file
for file_path in file_paths:
    # Read all lines from the file
    with open(file_path, 'r', errors='ignore') as f:
        lines = f.readlines()

    # Find the line index where data columns start (line after "(Deg)")
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('(Deg)'):
            data_start = i + 1
            break

    if data_start is None:
        print(f"Warning: Could not find data header '(Deg)' in {file_path}. Skipping...")
        continue

    try:
        # Load the data into a NumPy array
        data = np.loadtxt(file_path, skiprows=data_start)

        # Extract columns
        angles = data[:, 0]
        frequencies = data[:, 1]
        gains = data[:, 2]
        phases = data[:, 3]

        # Plot Gain vs. Aspect Angle for the specified frequency
        mask = frequencies == target_freq_mhz
        if not np.any(mask):
            print(f"Warning: No data found for frequency {FREQUENCY_GHZ} GHz in {file_path}. Skipping...")
            continue

        ang = angles[mask]
        gain_vals = gains[mask]
        phase_vals = phases[mask]
        # Sort by angle
        sort_idx = np.argsort(ang)
        ang_sorted = ang[sort_idx]
        gain_sorted = gain_vals[sort_idx]
        phase_sorted = phase_vals[sort_idx]
        
        # Apply azimuth offset
        if file_path == 'LL measurements/EAR_DISH_AZ_LONG_CUT_31_6_3_2025_064.std':
            az_offset = 0.0
        else:
            az_offset = 0.7
        ang_sorted = ang_sorted + az_offset
        
        # Calculate HPBW
        hpbw, left_cross, right_cross = find_hpbw(ang_sorted, gain_sorted)
        
        # Find max gain and sidelobes
        max_gain = np.max(gain_sorted)
        max_gain_angle = ang_sorted[np.argmax(gain_sorted)]
        sidelobes = find_sidelobes(ang_sorted, gain_sorted)
        
        plt.subplot(2, 1, 1)
        # Plot the gain pattern
        plt.plot(ang_sorted, gain_sorted, label=f'{file_path.split("/")[-1]}')
        
        # Plot HPBW markers
        # if hpbw is not None:
        #     plt.plot([left_cross, left_cross], [max_gain-3, max_gain-3-5], 'r--')
        #     plt.plot([right_cross, right_cross], [max_gain-3, max_gain-3-5], 'r--')
        #     # Add HPBW text annotation
        #     plt.annotate(f'HPBW = {hpbw:.2f}°', 
        #                 xy=((left_cross + right_cross)/2, max_gain-6),
        #                 xytext=(0, -15),
        #                 textcoords='offset points',
        #                 ha='center',
        #                 color='red',
        #                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # # Plot max gain marker and annotation
        # plt.plot(max_gain_angle, max_gain, 'go')
        # plt.annotate(f'Max Gain = {max_gain:.2f} dBi',
        #             xy=(max_gain_angle, max_gain),
        #             xytext=(0, 10),
        #             textcoords='offset points',
        #             ha='center',
        #             color='green',
        #             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Plot sidelobe markers and annotations
        # for angle, gain in sidelobes:
        #     plt.plot(angle, gain, 'mo')
        #     if angle < 0:
        #         text_offset = (-40, 10)
        #     else:
        #         text_offset = (40, 10)
        #     plt.annotate(f'Sidelobe = {gain:.2f} dBi',
        #                 xy=(angle, gain),
        #                 xytext=text_offset,
        #                 textcoords='offset points',
        #                 ha='center',
        #                 color='magenta',
        #                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

        # Plot phase pattern in the bottom subplot
        plt.subplot(2, 1, 2)
        plt.plot(ang_sorted, phase_sorted)
        plt.grid(True)
        plt.xlabel('Azimuth Angle [deg]')
        plt.ylabel('Phase [deg]')
        plt.xlim(-8, 8)
        # Set y-axis limits to show full phase range
        plt.ylim(-180, 180)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        continue

# Plot FEKO simulation data
# plt.subplot(2, 1, 1)
# plt.plot(feko_angles, feko_gains, '--', label='FEKO Simulation', color='gray', alpha=0.7)

# Configure top subplot (gain pattern)
plt.xlim(-8, 8)
plt.ylim(-20, 50)
plt.xlabel('Azimuth Angle [deg]')
plt.ylabel('Gain [dBi]')
plt.title(f'{FREQUENCY_GHZ} GHz')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
