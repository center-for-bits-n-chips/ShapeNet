import os
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
    # 'LL measurements/EAR_DISH_AZ_CUT_12_6_3_2025_042.std',
    #'LL measurements/EAR_DISH_EL_CUT_13_6_3_2025_044.std',
    #'LL measurements/EAR_DISH_EL_CUT_22_6_3_2025_055.std',

    ### THE BEST MEASUREMENTS
    ## Comparison of shaped and unshaped measurement
    'LL measurements/EAR_DISH_AZ_CUT_LONG_23_6_3_2025_056.std', # shaped
    #'LL measurements/EAR_DISH_AZ_LONG_CUT_31_6_3_2025_064.std', # unshaped

    ## Beam steering comparisons
    #'LL measurements/EAR_DISH_AZ_CUT_LONG_23_6_3_2025_056.std', # shaped
    #'LL measurements/EAR_DISH_AZ_CUT_34_6_3_2025_067.std', # positive shift
    #'LL measurements/EAR_DISH_AZ_CUT_37_6_3_2025_071.std', # negaative shift
    # 'LL measurements/EAR_DISH_AZ_CUT_35_6_3_2025_068.std', # extreme positive shift
    #'LL measurements/EAR_DISH_AZ_CUT_36_6_3_2025_069.std',
    #'LL measurements/EAR_DISH_AZ_CUT_36_6_3_2025_070.std',
]

# Per-file plot styles: color, alpha, and legend label
file_styles = {
    'LL measurements/EAR_DISH_AZ_CUT_LONG_23_6_3_2025_056.std': {'color': 'black',   'alpha': 1.0,  'label': 'Shaped'},
    'LL measurements/EAR_DISH_AZ_LONG_CUT_31_6_3_2025_064.std': {'color': 'black',   'alpha': 0.45, 'label': 'Unshaped'},
    'LL measurements/EAR_DISH_AZ_CUT_34_6_3_2025_067.std':      {'color': '#1f77b4', 'alpha': 0.45, 'label': None},  # positive shift - blue
    'LL measurements/EAR_DISH_AZ_CUT_37_6_3_2025_071.std':      {'color': '#d62728', 'alpha': 0.45, 'label': None},  # negative shift - red
    'LL measurements/EAR_DISH_AZ_CUT_35_6_3_2025_068.std':      {'color': '#2ca02c', 'alpha': 0.45, 'label': None},  # extreme positive - green
}

# Convert target frequency from GHz to MHz
target_freq_mhz = FREQUENCY_GHZ * 1000

# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
fig.subplots_adjust(hspace=0.15)

# Apply common tick style to both axes
for ax in (ax1, ax2):
    ax.tick_params(which='both', direction='in', top=True, right=True, bottom=True, left=True)

# Read FEKO simulation data
feko_data = np.loadtxt('measured pattern 35 GHz COMSOL f_D 1.3 case B.dat', skiprows=2)
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
        if file_path == 'LL measurements/EAR_DISH_AZ_LONG_CUT_31_6_3_2025_064.std' or \
            file_path == 'LL measurements/EAR_DISH_AZ_CUT_34_6_3_2025_067.std' or \
            file_path == 'LL measurements/EAR_DISH_AZ_CUT_35_6_3_2025_068.std' or \
            file_path == 'LL measurements/EAR_DISH_AZ_CUT_36_6_3_2025_069.std' or \
            file_path == 'LL measurements/EAR_DISH_AZ_CUT_36_6_3_2025_070.std' or \
            file_path == 'LL measurements/EAR_DISH_AZ_CUT_37_6_3_2025_071.std':
            az_offset = 0.0
        else:
            az_offset = 0.65
        ang_sorted = ang_sorted + az_offset
        
        # Calculate HPBW
        hpbw, left_cross, right_cross = find_hpbw(ang_sorted, gain_sorted)
        
        # Find max gain and sidelobes
        max_gain = np.max(gain_sorted)
        max_gain_angle = ang_sorted[np.argmax(gain_sorted)]
        sidelobes = find_sidelobes(ang_sorted, gain_sorted)
        
        style = file_styles.get(file_path, {'color': None, 'alpha': 1.0, 'label': None})

        # Plot the gain pattern
        ax1.plot(ang_sorted, gain_sorted,
                 color=style['color'], alpha=style['alpha'],
                 label=style.get('label'))

        # Annotate peak gain, HPBW, and worst sidelobe for labeled curves
        if style.get('label') is not None:
            color = style['color']
            alpha = style['alpha']

            # Peak gain marker + label
            ax1.plot(max_gain_angle, max_gain, 'o', color=color, alpha=alpha,
                     markersize=4)
            ax1.annotate(f"G = {max_gain:.1f} dBi",
                         xy=(max_gain_angle, max_gain),
                         xytext=(6, 2), textcoords='offset points',
                         fontsize=7, color=color, alpha=alpha,
                         ha='left', va='bottom')

            # HPBW horizontal indicator at half-power level + label
            if hpbw is not None:
                half_power = max_gain - 3
                ax1.plot([left_cross, right_cross], [half_power, half_power],
                         '--', color=color, alpha=alpha * 0.7, linewidth=1.0)
                ax1.annotate(f"HPBW = {hpbw:.2f}\u00b0",
                             xy=(right_cross, half_power),
                             xytext=(6, -2), textcoords='offset points',
                             fontsize=7, color=color, alpha=alpha,
                             ha='left', va='top')

            # Worst (highest) sidelobe marker + label
            if sidelobes:
                worst_sl_angle, worst_sl_gain = max(sidelobes, key=lambda x: x[1])
                sll_db = worst_sl_gain - max_gain
                ax1.plot(worst_sl_angle, worst_sl_gain, 'o', color=color, alpha=alpha,
                         markersize=4)
                ax1.annotate(f"SLL = {sll_db:.1f} dB",
                             xy=(worst_sl_angle, worst_sl_gain),
                             xytext=(6, 2), textcoords='offset points',
                             fontsize=7, color=color, alpha=alpha,
                             ha='left', va='bottom')

        angle_limit = 8

        # Configure top subplot (gain pattern)
        ax1.set_xlim(-angle_limit, angle_limit)
        ax1.set_ylim(-20, 50)
        ax1.set_ylabel('Gain [dBi]')

        # Plot phase pattern in the bottom subplot
        ax2.plot(ang_sorted, phase_sorted,
                 color=style['color'], alpha=style['alpha'])
        ax2.set_xlabel('Azimuth Angle [deg]')
        ax2.set_ylabel('Phase [deg]')
        ax2.set_xlim(-angle_limit, angle_limit)
        ax2.set_ylim(-200, 200)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        continue

# Plot FEKO simulation data on top subplot
#ax1.plot(feko_angles, feko_gains, '--', color='gray', alpha=0.7, label='FEKO Sim')

# Frequency label in top-left corner of top subplot
ax1.text(0.03, 0.97, f'{FREQUENCY_GHZ} GHz',
         transform=ax1.transAxes, va='top', ha='left', fontsize=9)

# Legend in top-right of gain subplot, no bounding box
ax1.legend(loc='upper right', frameon=False)

# Remove whitespace at the top of the figure
fig.subplots_adjust(top=0.99)

# Save figure as a PDF in a dedicated subfolder
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'radiation pattern figures')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'ka_{FREQUENCY_GHZ}_GHz_labeled.pdf')
fig.savefig(output_path, bbox_inches='tight', pad_inches=0.02)
print(f"Saved figure to {output_path}")

plt.show()
