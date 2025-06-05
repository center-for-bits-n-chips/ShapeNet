import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import itertools
import matplotlib.pyplot as plt

# 1. Load the CSV (semicolon-separated; fallback to latin1 if needed)
dic_folder = 'case B subset=125 step=41'
file_path = f'data/DIC/{dic_folder}/d0058.csv'  # your CSV
try:
    df = pd.read_csv(file_path, sep=';')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, sep=';', encoding='latin1')

# 2. Only use the first six columns (x, y, z, x_disp, y_disp, z_disp) and convert to meters
df6 = df.iloc[:, 0:6]
df6.columns = ['x', 'y', 'z', 'x_disp', 'y_disp', 'z_disp']
df6 = df6 / 1000.0  # Convert all measurements from mm to m

# 3. Extract arrays
x = df6['x'].to_numpy()
y = df6['y'].to_numpy()
z = df6['z'].to_numpy()
z_disp = df6['z_disp'].to_numpy()

# 4. Fit a circle to (x, y) to find the center (xc, yc) and original radius
A = np.column_stack((x, y, np.ones_like(x)))
b = x**2 + y**2
coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
xc = coeffs[0] / 2
yc = coeffs[1] / 2
radius_fit = np.sqrt(coeffs[2] + xc**2 + yc**2)
print(f"Fitted center: ({xc:.4f}, {yc:.4f}) m")
print(f"Fitted radius (original): {radius_fit:.4f} m")

# 5. Recenter interior points and compute deformed height
x_c = x - xc
y_c = y - yc
z_shape = z + z_disp

# Filter out points where z_shape > 0
mask = z_shape <= 0
x_c = x_c[mask]
y_c = y_c[mask]
z_shape = z_shape[mask]
print(f"Removed {len(x) - len(x_c)} points where z_shape > 0")

# 6. Create a boundary ring at r = 0.25 m (z = 0)
num_boundary = 180
angles = np.linspace(0, 2 * np.pi, num_boundary, endpoint=False)
xb = 0.25 * np.cos(angles)  # Changed from 250.0 to 0.25
yb = 0.25 * np.sin(angles)  # Changed from 250.0 to 0.25
zb = np.zeros_like(angles)

# 7. Combine for interpolation
interp_points = np.vstack((
    np.column_stack((x_c, y_c)),    # original, recentered
    np.column_stack((xb, yb))       # boundary ring
))
interp_values = np.concatenate((z_shape, zb))  # corresponding z values

# 8. Build a polar grid so triangle edges ≤ 0.0075 m
dr = 0.005  # radial step is 5 mm → diagonal ~7.07 mm
radii = np.arange(0, 0.25 + dr, dr)  # Changed from 250 to 0.25

# Generate (x,y) for each concentric circle
polars = []
for r in radii:
    if r == 0:
        polars.append((0.0, 0.0))
    else:
        # N_i points around circumference so spacing ~ dr
        Ni = int(np.ceil(2 * np.pi * r / dr))
        thetas = np.linspace(0, 2 * np.pi, Ni, endpoint=False)
        xs = r * np.cos(thetas)
        ys = r * np.sin(thetas)
        polars.extend(list(zip(xs, ys)))
polars = np.array(polars)
px = polars[:, 0]
py = polars[:, 1]

# 9. Interpolate z on each polar point (linear, boundary fill=0)
pz = griddata(
    points=interp_points,
    values=interp_values,
    xi=(px, py),
    method='linear',
    fill_value=0.0
)

# 10. Delaunay triangulation over (px, py)
points2D_ref = np.column_stack((px, py))
tri_ref = Delaunay(points2D_ref)
faces_ref = tri_ref.simplices

# 11. Compute max edge length of all triangles (should be < ~0.00707 m)
vertices_ref = np.column_stack((px, py, pz))
max_edge_len = 0.0
for tri_indices in faces_ref:
    pts = vertices_ref[tri_indices]
    for i, j in itertools.combinations(range(3), 2):
        d = np.linalg.norm(pts[i] - pts[j])
        if d > max_edge_len:
            max_edge_len = d
print(f"Max triangle edge length (polar refined): {max_edge_len:.4f} m")

# 12. Export an ASCII STL of the refined mesh
def compute_normal(v1, v2, v3):
    n = np.cross(v2 - v1, v3 - v1)
    norm = np.linalg.norm(n)
    return n / (norm if norm else 1.0)

output_stl_refined = f'export/{dic_folder}_mesh_refined.stl'
with open(output_stl_refined, 'w') as f:
    f.write('solid surface_refined\n')
    for tri_indices in faces_ref:
        p1, p2, p3 = vertices_ref[tri_indices]
        normal = compute_normal(p1, p2, p3)
        f.write(f'  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n')
        f.write('    outer loop\n')
        f.write(f'      vertex {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\n')
        f.write(f'      vertex {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}\n')
        f.write(f'      vertex {p3[0]:.6f} {p3[1]:.6f} {p3[2]:.6f}\n')
        f.write('    endloop\n')
        f.write('  endfacet\n')
    f.write('endsolid surface_refined\n')

print(f"Refined STL saved to: {output_stl_refined}")

# 13. (Optional) Visualize the refined mesh
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(
    px,
    py,
    pz,
    triangles=faces_ref,
    cmap='viridis',
    linewidth=0.2,
    antialiased=True
)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Polar-Refined Mesh (edge < 0.0075 m)')
plt.tight_layout()
plt.show()
