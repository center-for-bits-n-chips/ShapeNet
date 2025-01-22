import sys
import numpy as np
from NatNetClient import NatNetClient

import socket
import struct
import threading

import time

last_call_time = 0.0
elapsed_ms = 0.0

np.set_printoptions(linewidth=np.inf)

num_mesh_markers = 8
Z_center = 0.0

# NOTE that the marker IDs appear to change between optitrack power cycles
# however for the rigid bodies they are consistently 1,2,3
mesh_marker_positions = {}
ground_marker_positions = {marker_id: [0.0, 0.0, 0.0] for marker_id in [1, 2, 3]}
rim_marker_positions = {marker_id: [0.0, 0.0, 0.0] for marker_id in [1, 2, 3]}

def handle_client(conn, addr):
    global Z_center
    print(f"Connected by {addr}")
    #conn.settimeout(0.5)  # Set timeout to prevent blocking indefinitely
    try:
        while True:
            # Receive data from LabVIEW
            try:
                data = conn.recv(1024) # 1024 is the number of bytes
                if data:
                    # Process received data (assuming big-endian 4-byte float)
                    while len(data) >= 8:
                        num = struct.unpack('>d', data[:8])[0]
                        #print(f"Received from LabVIEW: {num}")
                        data = data[8:] # clear data out
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
            num_to_send = -Z_center  # negative to be consistent with laser reading
            #num_to_send = elapsed_ms
            data_to_send = struct.pack('>d', num_to_send)
            try:
                conn.sendall(data_to_send)
                #print(f"Sent to LabVIEW: {num_to_send}")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                print(f"Connection with {addr} was closed during send.")
            except Exception as e:
                print(f"Error sending data to {addr}: {e}")
    finally:
        conn.close()
        print(f"Connection with {addr} closed.")

def start_server(host, port):
    # Create a stop event
    #stop_event = threading.Event()
    #threads = []
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        #s.settimeout(1.0)  # Set timeout on socket operations
        print(f"Server listening on port {port}")
        try:
            while True:
                try:
                    conn, addr = s.accept()
                    handle_client(conn, addr)
                    print(f"Accepted connection from {addr}")
                    #server_thread = threading.Thread(target=handle_client, args=(stop_event, conn, addr))
                    #server_thread.start()
                    #threads.append(server_thread)
                except socket.timeout:
                    pass  # No connection attempt; check stop_event
        except KeyboardInterrupt:
            print("Server is shutting down.")
            s.close()
            print("All client connections have been closed.")

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
    # global last_call_time
    # global elapsed_ms
    global Z_center

    # current_time = time.perf_counter()

    # if last_call_time is not None:
    #     elapsed_ms = (current_time - last_call_time) * 1000.0
    #     #print(f"{elapsed_ms:.2f} ms since last function call")
    
    # last_call_time = current_time

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

    ground_marker_pos = list(ground_marker_positions.values())
    rim_marker_pos = list(rim_marker_positions.values())
    mesh_marker_pos = list(mesh_marker_positions.values())

    # Once all markers are updated, process the input
    all_rim_markers_nonzero = all(value != 0 for row in rim_marker_pos for value in row)
    all_mesh_markers_updated = len(mesh_marker_pos) == num_mesh_markers
    if all_rim_markers_nonzero and all_mesh_markers_updated:
        # draw a circle around the rim
        p1, p2, p3 = rim_marker_pos
        center, radius, normal = compute_circumradius(p1,p2,p3)
        
        # Target vector is the z-axis
        z_axis = np.array([0, 0, -1])
        
        # Compute rotation matrix
        rotation_mat = rotation_matrix_from_vectors(normal, z_axis)

        # Shift points over so center of rim is at (0,0,0)
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

        # CENTER LOCATION FROM MOCAP
        Z_mocap_mm = 1000 * input_array[2::3] # original array is in meters
        Z_center = Z_mocap_mm[3] # NOTE the center marker id changes

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
    streaming_client.set_print_level(0) # disables prints

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

    HOST = '127.0.0.1'  # Listen on specified network interface
    PORT = 9999         # Arbitrary non-privileged port
    start_server(HOST, PORT) # blocking

    streaming_client.shutdown()
    sys.exit(0)
        

if __name__ == "__main__":
    main()