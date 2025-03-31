import sys
from NatNet.NatNetClient import NatNetClient
import socket
import struct
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import time
from threading import Thread, Event

np.set_printoptions(linewidth=np.inf)

@dataclass
class MocapConfig:
    num_mesh_markers: int = 8
    rim_z_offset: float = -0.01754  # MEASURED WITH CALIPERS
    z_axis: np.ndarray = np.array([0, 0, -1])
    center: np.ndarray = np.array([0, 0, 0])
    display_update_rate: float = 10.0  # Hz

class DigitalDisplay:
    def __init__(self, mocap_server, update_rate: float = 10.0):
        self.mocap_server = mocap_server
        self.update_rate = update_rate
        self.stop_event = Event()
        self.display_thread = None
        self.last_positions = None
        # Move cursor to top-left and clear screen once at initialization
        print('\033[2J\033[H')

    def format_value(self, value: float) -> str:
        return f"{value:6.2f}"

    def display_positions(self):
        """Display the current mesh marker positions in a formatted table."""
        # Move cursor to home position (1,1)
        print('\033[H')
        print("\n=== Mesh Marker Positions (mm) ===")
        print("Marker ID |    X    |    Y    |    Z    |")
        print("-" * 40)
        
        if self.last_positions is None:
            print("Waiting for marker data...")
            # Clear any remaining lines
            print('\033[J')
            return

        for marker_id, pos in sorted(self.last_positions.items()):
            print(f"{marker_id:8d} | {self.format_value(pos[0])} | {self.format_value(pos[1])} | {self.format_value(pos[2])} |")
        
        print("-" * 40)
        print(f"Last update: {time.strftime('%H:%M:%S')}")
        print("\nPress Ctrl+C to exit")
        # Clear any remaining lines
        print('\033[J')

    def update_loop(self):
        """Main update loop for the display."""
        while not self.stop_event.is_set():
            self.display_positions()
            time.sleep(1.0 / self.update_rate)

    def start(self):
        """Start the display thread."""
        self.display_thread = Thread(target=self.update_loop)
        self.display_thread.daemon = True
        self.display_thread.start()

    def stop(self):
        """Stop the display thread."""
        self.stop_event.set()
        if self.display_thread:
            self.display_thread.join()

    def update_positions(self, positions: Dict[int, List[float]]):
        """Update the stored positions."""
        self.last_positions = positions

class MocapServer:
    def __init__(self, config: MocapConfig = MocapConfig()):
        self.config = config
        self.z_mocap_mm = [0.0] * config.num_mesh_markers
        self.mesh_marker_positions: Dict[int, List[float]] = {}
        self.rim_marker_positions: Dict[int, List[float]] = {marker_id: [0.0, 0.0, 0.0] for marker_id in [1, 2, 3]}
        self.normal = np.array([0, 0, -1])
        self.center = np.array([0, 0, 0])
        self.flag_calculate_rim = True
        self.display = DigitalDisplay(self, config.display_update_rate)

    def __del__(self):
        """Cleanup when the server is destroyed."""
        if hasattr(self, 'display'):
            self.display.stop()

    def rotation_matrix_from_vectors(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Returns the rotation matrix that aligns vector a to b."""
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        I = np.eye(3)
        
        if s == 0:
            return I

        vx = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        
        return I + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))

    def compute_circumradius(self, p1: List[float], p2: List[float], p3: List[float]) -> Tuple[np.ndarray, float, np.ndarray]:
        """Compute the circumradius, center, and normal of the circle through three points."""
        P1, P2, P3 = map(np.array, [p1, p2, p3])
        A, B = P2 - P1, P3 - P1
        cross_prod = np.cross(A, B)
        cross_prod_mag = np.linalg.norm(cross_prod)

        if np.isclose(cross_prod_mag, 0):
            raise ValueError("Points are collinear")

        normal = cross_prod / cross_prod_mag
        mid_AB, mid_AC = (P1 + P2) / 2, (P1 + P3) / 2
        perp_AB, perp_AC = np.cross(A, normal), np.cross(B, normal)

        matrix = np.vstack([perp_AB, -perp_AC]).T
        rhs = mid_AC - mid_AB
        t = np.linalg.lstsq(matrix, rhs, rcond=None)[0][0]
        
        center = mid_AB + t * perp_AB
        radius = np.linalg.norm(center - P1)
        return center, radius, normal

    def handle_client(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        """Handle interaction with a single connected client."""
        print(f"Connected by {addr}")
        conn.settimeout(0.1)

        try:
            while True:
                try:
                    data = conn.recv(1024)
                    if not data:
                        print(f"Client {addr} closed the connection.")
                        break
                    
                    while len(data) >= 8:
                        struct.unpack('>d', data[:8])[0]
                        data = data[8:]
                        
                    # Send data to client
                    format_string = '>' + 'd' * len(self.z_mocap_mm)
                    conn.sendall(struct.pack(format_string, *self.z_mocap_mm))
                    
                except socket.timeout:
                    continue
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
                    print(f"Connection error with {addr}: {e}")
                    break
        finally:
            conn.close()
            print(f"Connection with {addr} closed.\n\n\n")

    def receive_labeled_marker(self, marker_id: int, model_id: int, position: List[float]) -> None:
        """Handle labeled marker data from OptiTrack."""
        x, y, z = position
        if model_id == 0:  # mesh markers
            self.mesh_marker_positions[marker_id] = [y, z, x]
        elif model_id == 1:  # rim markers
            self.rim_marker_positions[marker_id] = [y, z, x]

        rim_marker_pos = list(self.rim_marker_positions.values())
        mesh_marker_pos = list(self.mesh_marker_positions.values())

        if not (all(value != 0 for row in rim_marker_pos for value in row) and 
                len(mesh_marker_pos) == self.config.num_mesh_markers):
            return

        if self.flag_calculate_rim:
            p1, p2, p3 = rim_marker_pos
            self.center, _, self.normal = self.compute_circumradius(p1, p2, p3)
            self.flag_calculate_rim = False
            print("Finished Calculating Rim")

        # Transform points
        rotation_mat = self.rotation_matrix_from_vectors(self.normal, self.config.z_axis)
        mesh_marker_pos = np.array(mesh_marker_pos) - self.center
        rim_marker_pos = np.array(rim_marker_pos) - self.center

        # Rotate points
        mesh_marker_pos = mesh_marker_pos @ rotation_mat.T
        rim_marker_pos = rim_marker_pos @ rotation_mat.T

        # Apply rim offset
        mesh_marker_pos[:, 2] += self.config.rim_z_offset

        # Update Z positions
        self.z_mocap_mm = [1000 * z for z in mesh_marker_pos[:, 2]]
        
        # Update display with current positions
        self.display.update_positions(dict(zip(range(len(mesh_marker_pos)), 1000.0 * mesh_marker_pos)))

    def start_server(self, host: str = '0.0.0.0', port: int = 9999) -> None:
        """Start the TCP server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen()
            s.settimeout(0.1)
            print(f"Server listening on port {port}")

            while True:
                try:
                    conn, addr = s.accept()
                    print(f"Accepted connection from {addr}")
                    self.handle_client(conn, addr)
                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    print("Server is shutting down.")
                    break

def main():
    # Parse command line arguments
    options = {
        "clientAddress": "127.0.0.1",
        "serverAddress": "127.0.0.1",
        "use_multicast": False,
        "enable_visualization": True,
        "enable_display": True  # New parameter to enable/disable digital display
    }
    
    if len(sys.argv) > 1:
        options["enable_visualization"] = sys.argv[1].lower() != "false"
    if len(sys.argv) > 2:
        options["enable_display"] = sys.argv[2].lower() != "false"

    # Initialize NatNet client
    streaming_client = NatNetClient()
    streaming_client.set_nat_net_version(4, 1)
    streaming_client.set_client_address(options["clientAddress"])
    streaming_client.set_server_address(options["serverAddress"])
    streaming_client.set_use_multicast(options["use_multicast"])
    streaming_client.set_print_level(0)

    # Create and initialize MocapServer
    mocap_server = MocapServer()
    streaming_client.labeled_marker_listener = mocap_server.receive_labeled_marker

    # Start the NatNet client
    if not streaming_client.run():
        print("ERROR: Could not start streaming client.")
        sys.exit(1)
    
    # Initialize and start the visualizer if enabled
    visualizer = None
    if options["enable_visualization"]:
        from mocap_visualizer import MoCapVisualizer
        visualizer = MoCapVisualizer(mocap_server)
        visualizer.start()
        print("3D visualization started")

    # Start the digital display if enabled
    if options["enable_display"]:
        mocap_server.display.start()
        print("Digital display started")

    # Start the server
    try:
        mocap_server.start_server()
    except KeyboardInterrupt:
        print("Main thread caught KeyboardInterrupt, shutting down server...")
    finally:
        # Clean shutdown
        if visualizer:
            visualizer.stop()
        mocap_server.display.stop()
        streaming_client.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()