import sys
import numpy as np
from NatNet.NatNetClient import NatNetClient
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import GLMeshItem
import torch
import torch.nn as nn
from torch.special import bessel_j0, bessel_j1
from scipy.special import jn_zeros, jv
from ShapeNet.ShapeNet import ShapeNet

import socket
import struct
import threading

#warnings.filterwarnings("ignore", category=RuntimeWarning)

np.set_printoptions(linewidth=np.inf)

refresh_rate_ms = 5
num_mesh_markers = 8
n_basis = 2 # number of basis functions

Z_center = 0.0
Z_mocap_mm = [0.0] * num_mesh_markers

def handle_client(stop_event, conn, addr):
    global Z_center
    global Z_mocap_mm
    print(f"Connected by {addr}")
    conn.settimeout(1.0)  # Set timeout to prevent blocking indefinitely
    try:
        while not stop_event.is_set():
            # Receive data from LabVIEW
            try:
                data = conn.recv(1024)
                if data:
                    # Process received data (assuming big-endian 4-byte float)
                    while len(data) >= 8:
                        num = struct.unpack('>d', data[:8])[0]
                        #print(f"Received from LabVIEW: {num}")
                        data = data[8:]
                else:
                    print(f"Client {addr} closed the connection.")
                    break  # Connection closed by client
            except socket.timeout:
                pass  # No data received; proceed to sending
            except (ConnectionResetError, ConnectionAbortedError):
                print(f"Connection with {addr} was reset during receive.")
                break
            except Exception as e:
                print(f"Error receiving data from {addr}: {e}")
                break

            # Send data to LabVIEW
            num_to_send = -Z_mocap_mm  # negative to be consistent with laser reading
            format_string = '>' + 'd' * len(Z_mocap_mm)
            data_to_send = struct.pack(format_string, *num_to_send)
            try:
                conn.sendall(data_to_send)
                #print(f"Sent to LabVIEW: {num_to_send}")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                print(f"Connection with {addr} was closed during send.")
                break
            except Exception as e:
                print(f"Error sending data to {addr}: {e}")
                break

    finally:
        conn.close()
        print(f"Connection with {addr} closed.")

def start_GUI(stop_event):
    while not stop_event.is_set():
        # Create the application instance
        app = QApplication(sys.argv)
        window1 = RealTimeMeshShape()
        window1.show()
        pg.exec()

def start_server(host, port):
    # Create a stop event
    stop_event = threading.Event()
    threads = []
    
    gui_thread = threading.Thread(target=start_GUI, args=(stop_event,))
    gui_thread.start()
    threads.append(gui_thread)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        s.settimeout(1.0)  # Set timeout on socket operations
        print(f"Server listening on port {port}")
        try:
            while not stop_event.is_set():
                try:
                    conn, addr = s.accept()
                    print(f"Accepted connection from {addr}")
                    server_thread = threading.Thread(target=handle_client, args=(stop_event, conn, addr))
                    server_thread.start()
                    threads.append(server_thread)
                    # If you want to handle only one connection, you can break here
                    # break
                except socket.timeout:
                    pass  # No connection attempt; check stop_event
        except KeyboardInterrupt:
            print("Main thread received KeyboardInterrupt. Stopping server...")
            stop_event.set()
        finally:
            print("Server is shutting down.")
            s.close()
            # Wait for all threads to finish
            for t in threads:
                t.join()
            print("All client connections have been closed.")

# JOHNDEBUG: THIS IS A BAD HACK RIGHT NOW
# USING BOTH A GLViewWidget and PlotWidget in the same class
# should separate into two classes and do the neural network processing separately
class RealTimeMeshShape(QMainWindow):
    def __init__(self, title="Real-Time Meshurements"):
        super().__init__()
        self.setWindowTitle(title)

        # Plot widget for the bar graph
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)
        
        init_data = range(num_mesh_markers)
        init_NN_output = range(3*n_basis)
        # Initialize the bar graph
        self.data = init_data
        self.mocap_bar_graph = pg.BarGraphItem(x=np.arange(num_mesh_markers), height=init_data, width=0.2, brush='r')
        self.NN_output_bar_graph = pg.BarGraphItem(x=np.arange(3*n_basis), height=init_NN_output, width=0.5, brush='g')
        self.plot_widget.addItem(self.mocap_bar_graph)
        self.plot_widget.addItem(self.NN_output_bar_graph)

        # Set up the OpenGL view widget
        self.view = gl.GLViewWidget()
        self.view.show()
        self.view.setWindowTitle('Real-time 3D Scatter Plot')
        self.view.setCameraPosition(distance=1)
        self.view.setGeometry(0, 110, 800, 600)

        # Add grid for reference
        self.grid = gl.GLGridItem()
        self.grid.scale(2, 2, 1)
        self.grid.setDepthValue(10)  # Ensure the grid is rendered below
        self.view.addItem(self.grid)

        # Initialize the scatter plot with dummy data
        pos = np.zeros((num_mesh_markers, 3))  # Starting positions
        self.scatter_ground = gl.GLScatterPlotItem(pos=pos, size=0.01, color=(1, 0, 0, 1), pxMode=False)
        self.scatter_rim = gl.GLScatterPlotItem(pos=pos, size=0.01, color=(0, 1, 0, 1), pxMode=False)
        self.scatter_mesh = gl.GLScatterPlotItem(pos=pos, size=0.01, color=(1, 1, 1, 1), pxMode=False)

        # Add axes for reference
        self.axes = gl.GLAxisItem()
        self.axes.setSize(x=10, y=10, z=10)
        self.view.addItem(self.axes)

        init_vertices = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=float)
            
        init_faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ], dtype=int)
            
        # Create MeshData
        meshdata = pg.opengl.MeshData(vertexes=init_vertices, faces=init_faces)

        # Create the mesh item
        self.mesh = GLMeshItem(
            meshdata = meshdata,
            smooth=False,
            color=(0.5, 0.5, 1, 1),  # RGBA
            shader='shaded',
            drawEdges=True,
        )

        # Add the mesh to the view
        self.view.addItem(self.mesh)

        circle_pts = np.zeros(3)

        # Create a GLLinePlotItem for the circle
        self.circle = gl.GLLinePlotItem(
            pos=circle_pts,
            color=(1, 0, 0, 1),  # Red color
            width=2,
            antialias=True
        )
        self.view.addItem(self.circle)

        #self.view.addItem(self.scatter_ground)
        self.view.addItem(self.scatter_rim)
        self.view.addItem(self.scatter_mesh)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(refresh_rate_ms)  # Update interval in milliseconds

    def update(self):
        global Z_center
        # Once all markers are updated, process the input
        if all(mesh_marker_positions.values()):
            # Update the scatter plot data
            ground_marker_pos = list(ground_marker_positions.values())
            rim_marker_pos = list(rim_marker_positions.values())
            mesh_marker_pos = list(mesh_marker_positions.values())

            # draw a circle around the rim
            p1, p2, p3 = rim_marker_pos

            center, radius, normal = compute_circumradius(p1,p2,p3)
            # TODO: change to be measured from markers
            gap = 0.035 # measured from physical mesh
            
            # Target vector is the z-axis
            z_axis = np.array([0, 0, -1])
            
            # Compute rotation matrix
            rotation_mat = rotation_matrix_from_vectors(normal, z_axis)

            # Generate circle points
            num_points = 100
            circle_pts = generate_circle(center, np.array([0,0,1]), radius, num_points)
            
            # Shift points over so center of rim is at (0,0,0)
            circle_pts = circle_pts - center
            mesh_marker_pos = mesh_marker_pos - center
            ground_marker_pos = ground_marker_pos - center
            rim_marker_pos = rim_marker_pos - center

            # Rotate all points so they are aligned with the z-axis rim
            rim_marker_pos = rotate_points(rim_marker_pos, rotation_mat)
            mesh_marker_pos = rotate_points(mesh_marker_pos, rotation_mat)

            # rim offset
            z_offset = -0.0195 #-0.0195 # 0.0038
            mesh_marker_pos[:,2] -= z_offset

            input_vector = []
            for marker in mesh_marker_pos:
                input_vector.extend(marker)
            input_array = np.array(input_vector)

            # normalize the input
            mocap_input = normalize_input(input_array, radius, gap)
            predicted_coefficients = process_input(mocap_input)
            # Print each number with 2 decimal places
            #print(" ".join(f"{num:.2f}" for num in shape_net_input[2::3]))

            # compute reconstructed shape from NN output            
            r_full = torch.linspace(0, 1.0, 50)
            theta_full = torch.linspace(0, 2*np.pi, 50)
            Theta, R = torch.meshgrid(theta_full, r_full, indexing='ij')

            full_basis_functions = generate_basis_functions_for_surface(R, Theta, n_basis).detach().numpy()
            Z_np = np.dot(full_basis_functions, predicted_coefficients)

            # CENTER LOCATION FROM NN
            #center_basis_functions = generate_basis_functions_for_surface(torch.zeros(1, 1),torch.zeros(1, 1), n_basis).detach().numpy()
            #Z_center_np = np.dot(center_basis_functions, predicted_coefficients)
            #Z_center = Z_center_np.item()

            # CENTER LOCATION FROM MOCAP
            Z_mocap_mm = 1000 * input_array[2::3] # original array is in meters
            Z_center = Z_mocap_mm[3] # NOTE the center marker id changes

            # redimensionalize
            R = radius * R
            Z_np = Z_np*gap

            X = R*torch.cos(Theta)
            Y = R*torch.sin(Theta)
            X_np = X.detach().numpy()
            Y_np = Y.detach().numpy()

            # increase deformation scaling
            #Z_scaling = 1.0 / gap # scaling for 0 to 1 gap
            Z_scaling = 10.0
            Z_np *= Z_scaling
            mesh_marker_pos[:,2] *= Z_scaling
            
            # Example: Color based on Z-value
            colors = np.zeros((X_np.size, 4), dtype=float)
            colors[:, 0] = (Z_np.flatten() - Z_np.min()) / (Z_np.max() - Z_np.min())  # Red channel
            colors[:, 1] = 0.5  # Green channel
            colors[:, 2] = 1 - (Z_np.flatten() - Z_np.min()) / (Z_np.max() - Z_np.min())  # Blue channel
            colors[:, 3] = 1.0  # Alpha channel

            # Flatten the meshgrid arrays to create a list of vertices
            vertices = np.column_stack((X_np.flatten(), Y_np.flatten(), Z_np.flatten()))
            # Create faces
            faces = create_faces(X_np, Y_np)

            meshdata = gl.MeshData(vertexes = vertices, faces = faces)
            meshdata.setVertexColors(colors)
            self.mesh.setMeshData(meshdata = meshdata)

            self.circle.setData(pos=circle_pts)
            self.scatter_ground.setData(pos=ground_marker_pos)
            self.scatter_rim.setData(pos=rim_marker_pos)
            self.scatter_mesh.setData(pos=mesh_marker_pos)

            #self.data = Z_mocap_mm # plot the z-displacement in mm
            self.data = mocap_input[2::3] # z-input to shape net
            self.mocap_bar_graph.setOpts(height=self.data)
            self.NN_output_bar_graph.setOpts(height=predicted_coefficients)

# Path to the saved .pth file
# model_path = 'shape_net_model_2_basis_8_markers_new.pth' # simulated data
model_path = 'shape_net_model.pth'

# Load the state_dict
state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)  # Use 'cuda' if using GPU

# Define Model & Training Parameters
input_size = 3*num_mesh_markers+1  # Number of features in X_train
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

# NOTE that the marker IDs appear to change between optitrack power cycles
# however for the rigid bodies they are consistently 1,2,3
mesh_marker_positions = {}
ground_marker_positions = {marker_id: (0.0, 0.0, 0.0) for marker_id in [1, 2, 3]}
rim_marker_positions = {marker_id: (0.0, 0.0, 0.0) for marker_id in [1, 2, 3]}

def normalize(v):
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    return v  # Return the original vector if it's already a zero or near-zero vector

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

    return np.array(faces, dtype=int)

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

def rotation_matrix_from_vectors(a, b):
    """ Returns the rotation matrix that aligns a to b
    :param a: A 3d "source" vector
    :param b: A 3d "destination" vector
    :return mat: A transformation matrix (3x3) which rotates vec1 to vec2
    """
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    I = np.eye(3)
    
    if s == 0:  # vec1 and vec2 are already aligned
        return I

    vx = np.array([[0, -v[2], v[1]], 
                   [v[2], 0, -v[0]], 
                   [-v[1], v[0], 0]])
    
    rotation_matrix = I + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    return rotation_matrix

# Apply rotation to all points
def rotate_points(points, rotation_mat):
    return points @ rotation_mat.T

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

# Callback function to handle labeled marker data
def receive_labeled_marker(marker_id, model_id, position):
    if model_id == 0: # mesh markers
        x, y, z = position # rescramble position
        reoriented_position = [y, z, x]
        mesh_marker_positions[marker_id] = reoriented_position
    if model_id == 1: # ground markers
        x, y, z = position # rescramble position
        reoriented_position = [y, z, x]
        ground_marker_positions[marker_id] = reoriented_position
    if model_id == 2: # rim markers
        x, y, z = position # rescramble position
        reoriented_position = [y, z, x]
        rim_marker_positions[marker_id] = reoriented_position

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

def normalize_input(input, radius, gap):
    # (x,y,z)
    # note that x is the mesh displacement
    # redefine x,y,z to match neural network input

    x = input[0::3]
    y = input[1::3]
    z = input[2::3]

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
    streaming_client.set_nat_net_version(4,1)
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

    try:
        HOST = '0.0.0.0'  # Listen on specified network interface
        PORT = 9999         # Arbitrary non-privileged port
        start_server(HOST, PORT)
    except KeyboardInterrupt:
        streaming_client.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()