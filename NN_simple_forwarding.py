import sys
import numpy as np
from NatNetClient import NatNetClient

import socket
import struct

import torch
from ShapeNet import ShapeNet
from torch.special import bessel_j0, bessel_j1
#from scipy.special import jn_zeros, jv

import time

np.set_printoptions(linewidth=np.inf)

num_mesh_markers = 8
Z_center = 0.0
Z_mocap_mm = [0.0]*(num_mesh_markers + 1) # extra marker is the shapenet center
n_basis = 2 # number of basis functions
gap = 37 # [mm]
rim_z_offset = 0.01754 # MEASURED WITH CALIPERS

# Create Bessel Function Zeros Table
# alphas = []
# for m in range(2):
#     zeros = jn_zeros(m, n_basis)
#     alphas.extend(zeros)

alphas = [2.404825557695773, 5.520078110286311, 3.831705970207512, 7.015586669815619]

# NOTE that the marker IDs appear to change between optitrack power cycles
# however for the rigid bodies they are consistently 1,2,3
mesh_marker_positions = {}
rim_marker_positions = {marker_id: [0.0, 0.0, 0.0] for marker_id in [1, 2, 3]}
normal = np.array([0, 0, -1])
center = np.array([0, 0, 0])
input_array = np.zeros(3*num_mesh_markers)
radius = 0.25 # 25 cm
flag_calculate_rim = True

# Path to the saved .pth file
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
model.eval()

# Placeholder function for neural network processing
def process_input(input):
    # Replace this with your neural network inference code
    voltage = [0.0]
    voltage.extend(input)
    voltage = np.array(voltage, dtype=np.float32) # NN input must be float

    full_input = torch.from_numpy(voltage)
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        predicted_coefficients = model(full_input).numpy().flatten()
    
    return predicted_coefficients

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

def normalize_input(input, radius, gap):
    # (x,y,z)
    # note that x is the mesh displacement
    # redefine x,y,z to match neural network input

    x = input[0::3]
    y = input[1::3]
    z = input[2::3]

    r, theta = cartesian_to_polar_numpy(x, y)
    r_normalized = r / radius
    z_normalized = 1000*z / gap # covert z from meters to mm

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

def handle_client(conn, addr):
    """
    Handle interaction with a single connected client.
    """
    global Z_mocap_mm, input_array, radius
    print(f"Connected by {addr}")

    # Prevent blocking forever when calling conn.recv()
    conn.settimeout(0.1)

    try:
        while True:
            # Receive data from LabVIEW
            try:
                data = conn.recv(1024)  # 1024 is the number of bytes
                if data:
                    while len(data) >= 8:
                        num = struct.unpack('>d', data[:8])[0]
                        data = data[8:]
                else:
                    print(f"Client {addr} closed the connection.")
                    break
            except socket.timeout:
                pass
            except (ConnectionResetError, ConnectionAbortedError):
                print(f"Connection with {addr} was reset during receive.")
                break
            except Exception as e:
                print(f"Error receiving data from {addr}: {e}")
                break

            # Calculate NN
            # normalize the input
            mocap_input = normalize_input(input_array, radius, gap)
            print(mocap_input)
            predicted_coefficients = process_input(mocap_input)
      
            # CENTER LOCATION FROM NN
            center_basis_functions = generate_basis_functions_for_surface(torch.zeros(1, 1),torch.zeros(1, 1), n_basis).detach().numpy()
            Z_center_np = np.dot(center_basis_functions, predicted_coefficients)
            Z_center_shapenet = Z_center_np.item()*gap
            Z_mocap_mm[0] = Z_center_shapenet
            
            # Send data to LabVIEW
            num_to_send = Z_mocap_mm
            format_string = '>' + 'd' * len(Z_mocap_mm)
            data_to_send = struct.pack(format_string, *num_to_send)
            try:
                conn.sendall(data_to_send)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                print(f"Connection with {addr} was closed during send.")
                break
            except Exception as e:
                print(f"Error sending data to {addr}: {e}")
                break
    finally:
        conn.close()
        print(f"Connection with {addr} closed.")

def start_server(host, port):
    """
    Create and run the TCP server. This function blocks until stopped (e.g., via Ctrl+C).
    """
    # Use a context manager so the socket is cleaned up automatically
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        # Also give the server socket a timeout so we can detect Ctrl+C in accept().
        s.settimeout(0.1)
        print(f"Server listening on port {port}")

        # We keep accepting connections until KeyboardInterrupt is raised.
        while True:
            try:
                conn, addr = s.accept()
                print(f"Accepted connection from {addr}")
                handle_client(conn, addr)
            except socket.timeout:
                pass  # No incoming connections, continue
            except KeyboardInterrupt:
                # Once we catch KeyboardInterrupt, break out of the loop
                print("Server is shutting down.")
                s.close()
                break

    print("All client connections have been closed.")

def rotation_matrix_from_vectors(a, b):
    """
    Returns the rotation matrix that aligns vector a to b.
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

def rotate_points(points, rotation_mat):
    """
    Apply a rotation (3x3) to an array of points.
    """
    return points @ rotation_mat.T

def compute_circumradius(p1, p2, p3):
    """
    Compute the circumradius, center, and normal of the unique circle
    passing through three non-collinear points in 3D space.
    """
    # Convert points to NumPy arrays
    P1 = np.array(p1, dtype=float)
    P2 = np.array(p2, dtype=float)
    P3 = np.array(p3, dtype=float)

    # Compute vectors
    A = P2 - P1
    B = P3 - P1

    cross_prod = np.cross(A, B)
    cross_prod_mag = np.linalg.norm(cross_prod)

    if np.isclose(cross_prod_mag, 0):
        raise ValueError("The three points are collinear; no unique circumcircle exists.")

    normal = cross_prod / cross_prod_mag

    # Compute midpoints
    mid_AB = (P1 + P2) / 2
    mid_AC = (P1 + P3) / 2

    # Perpendicular vectors in the plane
    perp_AB = np.cross(A, normal)
    perp_AC = np.cross(B, normal)

    # Solve for intersection of lines
    matrix = np.vstack([perp_AB, -perp_AC]).T
    rhs = mid_AC - mid_AB

    ts, residuals, rank, s = np.linalg.lstsq(matrix, rhs, rcond=None)
    t, _ = ts
    center = mid_AB + t * perp_AB
    radius = np.linalg.norm(center - P1)

    return center, radius, normal

# Callback function to handle labeled marker data
def receive_labeled_marker(marker_id, model_id, position):
    global Z_mocap_mm, flag_calculate_rim, center, normal, input_array, radius
    
    if model_id == 0:  # mesh markers
        x, y, z = position
        mesh_marker_positions[marker_id] = [y, z, x]
    elif model_id == 1:  # rim markers
        x, y, z = position
        rim_marker_positions[marker_id] = [y, z, x]

    rim_marker_pos = list(rim_marker_positions.values())
    mesh_marker_pos = list(mesh_marker_positions.values())

    all_rim_markers_nonzero = all(value != 0 for row in rim_marker_pos for value in row)
    all_mesh_markers_updated = (len(mesh_marker_pos) == num_mesh_markers)

    if all_rim_markers_nonzero and all_mesh_markers_updated:
        if flag_calculate_rim:
            # draw a circle around the rim
            p1, p2, p3 = rim_marker_pos
            center, radius, normal = compute_circumradius(p1,p2,p3)
            flag_calculate_rim = False
            print("Finished Calculating Rim")
        
        # Target vector is the z-axis
        z_axis = np.array([0, 0, -1])
        
        # Compute rotation matrix
        rotation_mat = rotation_matrix_from_vectors(normal, z_axis)

        # Shift points over so center of rim is at (0,0,0)
        mesh_marker_pos = np.array(mesh_marker_pos) - center
        rim_marker_pos = np.array(rim_marker_pos) - center

        # Rotate all points so they are aligned with the z-axis rim
        rim_marker_pos = rotate_points(rim_marker_pos, rotation_mat)
        mesh_marker_pos = rotate_points(mesh_marker_pos, rotation_mat)

        # correct for rim offset
        mesh_marker_pos[:,2] += rim_z_offset

        input_vector = []
        for marker in mesh_marker_pos:
            input_vector.extend(marker)
        input_array = np.array(input_vector)
        
        # CENTER LOCATION FROM MOCAP (in mm)
        # reverse sign to make it positive for the laser
        Z_mocap_mm[1:] = -1000 * input_array[2::3]  # original array is in meters

def my_parse_args(arg_list, args_dict):
    # set up base values
    arg_list_len = len(arg_list)
    if arg_list_len > 1:
        args_dict["serverAddress"] = arg_list[1]
        if arg_list_len > 2:
            args_dict["clientAddress"] = arg_list[2]
        if arg_list_len > 3:
            if len(arg_list[3]):
                args_dict["use_multicast"] = True
                if arg_list[3][0].upper() == "U":
                    args_dict["use_multicast"] = False

    return args_dict

def main():
    optionsDict = {}
    optionsDict["clientAddress"] = "127.0.0.1"
    optionsDict["serverAddress"] = "127.0.0.1"
    optionsDict["use_multicast"] = False

    # Parse any command-line arguments
    optionsDict = my_parse_args(sys.argv, optionsDict)

    # Create a new NatNet client
    streaming_client = NatNetClient()
    streaming_client.set_nat_net_version(4, 1)
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])
    streaming_client.set_print_level(0)  # disables prints

    # Set callbacks
    streaming_client.labeled_marker_listener = receive_labeled_marker

    is_running = streaming_client.run()
    if not is_running:
        print("ERROR: Could not start streaming client.")
        sys.exit(1)

    HOST = '0.0.0.0'    # Listen on all network interfaces
    PORT = 9999         # Arbitrary non-privileged port

    try:
        # This call blocks until KeyboardInterrupt
        start_server(HOST, PORT)
    except KeyboardInterrupt:
        print("Main thread caught KeyboardInterrupt, shutting down server...")

    # Cleanly shut down the streaming client
    streaming_client.shutdown()
    sys.exit(0)

if __name__ == "__main__":
    main()
