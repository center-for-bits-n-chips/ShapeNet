import sys
import time
import numpy as np
from NatNetClient import NatNetClient
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import GLMeshItem
import torch
import torch.nn as nn
from torch.special import bessel_j0, bessel_j1
from scipy.special import jn_zeros, jv

# Define the neural network model
class ShapeNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ShapeNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size) # Output layer
        self.dropout = nn.Dropout(p=0.2)  # Dropout with 20% probability

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

# Path to the saved .pth file
model_path = 'shape_net_model.pth'

# Load the state_dict
state_dict = torch.load(model_path, map_location=torch.device('cpu'))  # Use 'cuda' if using GPU

n_basis = 3
n_samples = 7

# Define Model & Training Parameters
input_size = 3*n_samples+1  # Number of features in X_train
hidden_size = 128
output_size = 3*n_basis  # Number of basis function coefficients

# Initialize the neural network
model = ShapeNet(input_size, hidden_size, output_size)

# Load the state_dict into the model
model.load_state_dict(state_dict)

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set model to evaluation mode
model.eval()

# Create the application instance
app = QtWidgets.QApplication([])

# Set up the OpenGL view widget
view = gl.GLViewWidget()
view.show()
view.setWindowTitle('Real-time 3D Scatter Plot')
view.setCameraPosition(distance=1)
view.setGeometry(0, 110, 800, 600)

# Add grid for reference
grid = gl.GLGridItem()
grid.scale(2, 2, 1)
grid.setDepthValue(10)  # Ensure the grid is rendered below
view.addItem(grid)

# Initialize the scatter plot with dummy data
num_points = 13  # Number of points to plot
pos = np.zeros((num_points, 3))  # Starting positions
scatter_ground = gl.GLScatterPlotItem(pos=pos, size=0.01, color=(1, 0, 0, 1), pxMode=False)
scatter_rim = gl.GLScatterPlotItem(pos=pos, size=0.01, color=(0, 1, 0, 1), pxMode=False)
scatter_mesh = gl.GLScatterPlotItem(pos=pos, size=0.01, color=(1, 1, 1, 1), pxMode=False)

view.addItem(scatter_ground)
view.addItem(scatter_rim)
view.addItem(scatter_mesh)

# Function to generate circle points in 3D
def generate_circle(center, normal, radius, num_points):
    # Create orthogonal vectors u and v in the plane
    if np.allclose(normal, [0.0, 0.0, 1.0]):
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.array([0.0, 0.0, 1.0])
    u = u - u.dot(normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    # Generate circle points
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_points = center[:, np.newaxis] + radius * (np.outer(u, np.cos(theta)) + np.outer(v, np.sin(theta)))
    return circle_points.T  # Shape: (num_points, 3)

# Add axes for reference
axes = gl.GLAxisItem()
axes.setSize(x=10, y=10, z=10)
view.addItem(axes)

def create_faces(X, Y):
    """
    Create a list of triangle faces from meshgrid coordinates.
    """
    faces = []
    rows, cols = X.shape

    for i in range(rows - 1):
        for j in range(cols - 1):
            # Define the indices of the square's corners
            idx0 = i * cols + j
            idx1 = idx0 + 1
            idx2 = idx0 + cols
            idx3 = idx2 + 1

            # First triangle of the square
            faces.append([idx0, idx2, idx1])

            # Second triangle of the square
            faces.append([idx1, idx2, idx3])

    return np.array(faces)

# Create Bessel Function Zeros Table
alphas = []
for m in range(2):
    zeros = jn_zeros(m, n_basis)
    alphas.extend(zeros)

# Bessel functions
def generate_basis_functions_for_surface(r, theta, N):
    basis_functions = []
    for k in range(N):
        n = k + 1 # indexing by n = 1, 2, 3
        phi_n = bessel_j0(alphas[k]*r)
        phi_n_sin = bessel_j1(alphas[n_basis + k]*r) * torch.sin(theta)
        phi_n_cos = bessel_j1(alphas[n_basis + k]*r) * torch.cos(theta)
        basis_functions.append(phi_n)
        basis_functions.append(phi_n_sin)
        basis_functions.append(phi_n_cos)
    return torch.stack(basis_functions, dim=2)  # Shape: (n_samples, n_basis)

# Update function
def update():
    # Update the scatter plot data
    ground_marker_pos = []
    rim_marker_pos = []
    mesh_marker_pos = []
    for marker_id in mesh_marker_ids:
        mesh_marker_pos.append(list(marker_positions[marker_id]))
    for marker_id in rim_marker_ids:
        rim_marker_pos.append(list(marker_positions[marker_id]))
    for marker_id in ground_marker_ids:
        ground_marker_pos.append(list(marker_positions[marker_id]))

    # draw a circle around the rim
    p1, p2, p3 = rim_marker_pos

    center, radius, normal = compute_circumradius(p1,p2,p3)

    # Generate circle points
    num_points = 100
    circle_pts = generate_circle(center, normal, radius, num_points)

    # Create a GLLinePlotItem for the circle
    circle = gl.GLLinePlotItem(
        pos=circle_pts,
        color=(1, 0, 0, 1),  # Red color
        width=2,
        antialias=True
    )
    view.addItem(circle)

    # Once all markers are updated, process the input
    if all(marker_positions.values()):
        input_vector = []
        for marker_id in mesh_marker_ids:
            input_vector.extend(marker_positions[marker_id])
        input_array = np.array(input_vector)

    # normalize the input
    shape_net_input = normalize_input(input_array, radius, center)
    predicted_coefficients = process_input(shape_net_input)

    r_full = torch.linspace(0, 1, 20)
    theta_full = torch.linspace(0, 2 * np.pi, 20)
    Theta, R = torch.meshgrid(theta_full, r_full, indexing='ij')

    full_basis_functions = generate_basis_functions_for_surface(R, Theta, n_basis).detach().numpy()
    Z = np.dot(full_basis_functions, predicted_coefficients)

    # redimensionalize
    gap = 0.035
    R = radius * R
    Z = Z

    X = R*torch.cos(Theta)
    Y = R*torch.sin(Theta)
    X = X.detach().numpy()
    Y = Y.detach().numpy()

    # shift back to center
    x_offset = center[0]
    y_offset = center[1]
    z_offset = center[2]

    X = X + x_offset
    Y = Y + y_offset
    Z = Z + z_offset

    # Example: Color based on Z-value
    colors = np.zeros((X.size, 4), dtype=float)
    colors[:, 0] = (Z.flatten() - Z.min()) / (Z.max() - Z.min())  # Red channel
    colors[:, 1] = 0.5  # Green channel
    colors[:, 2] = 1 - (Z.flatten() - Z.min()) / (Z.max() - Z.min())  # Blue channel
    colors[:, 3] = 1.0  # Alpha channel

    # Flatten the meshgrid arrays to create a list of vertices
    vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    # Create faces
    faces = create_faces(X, Y)

    
    # Create the mesh item
    mesh = GLMeshItem(
        vertexes=vertices,
        faces=faces,
        faceColors=colors,  # Optional: use colors
        smooth=True,
        drawEdges=False,
        computeNormals=True
    )

    # Add the mesh to the view
    view.addItem(mesh)

    scatter_ground.setData(pos=ground_marker_pos)
    scatter_rim.setData(pos=rim_marker_pos)
    scatter_mesh.setData(pos=mesh_marker_pos)

def compute_circumradius(p1, p2, p3):
    """
    Compute the circumradius, center, and normal of the unique circle passing through three non-collinear points in 3D space.

    Parameters:
    - p1, p2, p3: Tuples or lists representing the (x, y, z) coordinates of the three points.

    Returns:
    - center: A NumPy array of shape (3,) representing the (x, y, z) coordinates of the circumcircle's center.
    - radius: A float representing the radius of the circumcircle.
    - normal: A NumPy array of shape (3,) representing the normal vector of the circumcircle's plane.

    Raises:
    - ValueError: If the three points are collinear.
    """
    # Convert points to NumPy arrays
    P1 = np.array(p1, dtype=float)
    P2 = np.array(p2, dtype=float)
    P3 = np.array(p3, dtype=float)

    # Compute vectors A and B
    A = P2 - P1
    B = P3 - P1

    # Compute the cross product of A and B
    cross_prod = np.cross(A, B)
    cross_prod_mag = np.linalg.norm(cross_prod)

    # Check for collinearity
    if np.isclose(cross_prod_mag, 0):
        raise ValueError("The three points are collinear; no unique circumcircle exists.")

    # Compute the normal vector (unit vector)
    normal = cross_prod / cross_prod_mag

    # Compute midpoints of segments P1P2 and P1P3
    mid_AB = (P1 + P2) / 2
    mid_AC = (P1 + P3) / 2

    # Compute the direction vectors perpendicular to A and B in the plane
    perp_AB = np.cross(A, normal)
    perp_AC = np.cross(B, normal)

    # Set up the system of linear equations to find the intersection point (center)
    # mid_AB + t * perp_AB = mid_AC + s * perp_AC
    # This can be rearranged to: t * perp_AB - s * perp_AC = mid_AC - mid_AB
    matrix = np.vstack([perp_AB, -perp_AC]).T
    rhs = mid_AC - mid_AB

    # Solve for t and s using least squares
    try:
        ts, residuals, rank, s = np.linalg.lstsq(matrix, rhs, rcond=None)
        t, _ = ts
    except np.linalg.LinAlgError:
        raise ValueError("Failed to compute circumcircle center due to numerical issues.")

    # Compute the circumcircle center
    center = mid_AB + t * perp_AB

    # Compute the radius
    radius = np.linalg.norm(center - P1)

    return center, radius, normal

# Set up a timer for periodic updates
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(5)  # Update interval in milliseconds

# List of marker IDs for the markers you're tracking
# NOTE that the marker IDs appear to change between optitrack power cycles
desired_marker_ids = [2874, 3703, 3790, 3901, 4877, 4879, 4880, 4881, 4882, 4883, 4884, 7659, 7672]
mesh_marker_ids = [3703, 3790, 3901, 4880, 4882,  4884, 7672]
rim_marker_ids = [2874, 7659, 4883]
ground_marker_ids = [4881, 4877, 4879] # 4879 is origin (0,0,0)

# Dictionary to store marker positions
marker_positions = {marker_id: (0.0, 0.0, 0.0) for marker_id in desired_marker_ids}

# Callback function to handle labeled marker data
def receive_labeled_marker(marker_id, position):
    if marker_id in desired_marker_ids:
        x, y, z = position # rescramble position
        reoriented_position = [y, z, x]
        marker_positions[marker_id] = reoriented_position

# Placeholder function for neural network processing
def process_input(input):
    # Replace this with your neural network inference code
    q = [0.0]
    q.extend(input)
    q = np.array(q, dtype=np.float32) # NN input must be float

    full_input = torch.from_numpy(q)
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        predicted_coefficients = model(full_input).numpy().flatten()
    
    return predicted_coefficients

def normalize_input(input, radius, center):
    # (x,y,z)
    # note that x is the mesh displacement
    # redefine x,y,z to match neural network input
    gap = 0.035 # 3.5 cm

    x = input[0::3]
    y = input[1::3]
    z = input[2::3]
    
    x_offset = center[0]
    y_offset = center[1]
    z_offset = center[2]

    x = x - x_offset
    y = y - y_offset
    z = z - z_offset

    r, theta = cartesian_to_polar_numpy(x, y)
    r_normalized = r / radius
    z_normalized = z / gap

    # Use zip and list comprehension to interleave
    normalized_input = [item for trio in zip(r_normalized, theta, z_normalized) for item in trio]
    return normalized_input

def cartesian_to_polar_numpy(x, y):
    """
    Convert lists or NumPy arrays of Cartesian coordinates (x, y) to Polar coordinates (r, theta).
    
    Parameters:
    - x (list or np.ndarray): X-coordinates.
    - y (list or np.ndarray): Y-coordinates.
    
    Returns:
    - r (np.ndarray): Radial distances.
    - theta (np.ndarray): Angles in radians.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    r = np.hypot(x, y)          # Equivalent to sqrt(x**2 + y**2)
    theta = np.arctan2(y, x) + np.pi    # Angle in radians between -pi and pi
    
    return r, theta

def my_parse_args(arg_list, args_dict):
    # set up base values
    arg_list_len=len(arg_list)
    if arg_list_len>1:
        args_dict["serverAddress"] = arg_list[1]
        if arg_list_len>2:
            args_dict["clientAddress"] = arg_list[2]
        if arg_list_len>3:
            if len(arg_list[3]):
                args_dict["use_multicast"] = True
                if arg_list[3][0].upper() == "U":
                    args_dict["use_multicast"] = False

    return args_dict

# Main function to run the client
def main():

    optionsDict = {}
    optionsDict["clientAddress"] = "127.0.0.1"
    optionsDict["serverAddress"] = "127.0.0.1"
    optionsDict["use_multicast"] = False

    # This will create a new NatNet client
    optionsDict = my_parse_args(sys.argv, optionsDict)

    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])
    streaming_client.set_print_level(0) # print every 120 frames (1 Hz)

    # Set callbacks
    streaming_client.labeled_marker_listener = receive_labeled_marker

    is_running = streaming_client.run()
    if not is_running:
        print("ERROR: Could not start streaming client.")
        try:
            sys.exit(1)
        except SystemExit:
            print("...")
        finally:
            print("exiting")

    if streaming_client.connected() is False:
        print("ERROR: Could not connect properly.  Check that Motive streaming is on.")
        try:
            sys.exit(2)
        except SystemExit:
            print("...")
        finally:
            print("exiting")

    pg.exec()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        streaming_client.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()