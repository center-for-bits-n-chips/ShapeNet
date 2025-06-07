from __future__ import annotations
import sys
from NatNet.NatNetClient import NatNetClient
import socket
import struct
import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
from threading import Thread, Event
from digital_display import DigitalDisplay

import torch
from ShapeNet.ShapeNet import ShapeNet
from torch.special import bessel_j0, bessel_j1
from scipy.special import jn, jn_zeros
np.set_printoptions(linewidth=np.inf)

@dataclass
class MocapConfig:
    num_mesh_markers: int = 14
    z_axis: np.ndarray = np.array([0, 0, -1])
    center_m: np.ndarray = np.array([0, 0, 0])
    display_update_rate: float = 10.0  # Hz
    tare: bool = True  # Whether to subtract the initial offset
    n_basis: int = 1 # number of angular zeros
    m_basis: int = 2 # number of radial zeros
    tikhonov_weight: float = 1e-2
    model_path: str = 'case A shape_net_model.pth'
    hidden_size: int = 32
    NN_enable: bool = False
    LS_enable: bool = True

@dataclass
class Mode:
    """Single (n, m) Bessel mode with either cos(nθ) or sin(nθ) angular part."""

    n: int               # angular order
    k: float             # k_{n,m} satisfying J_n(k) = 0
    kind: str            # 'cos' or 'sin'
    omega: float         # energy weighting (k^2 for tensioned membrane)

    def phi(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Evaluate this mode at polar coords (r, θ)."""
        base = jn(self.n, self.k * r)
        if self.kind == "cos":
            return base * np.cos(self.n * theta)
        else:
            return base * np.sin(self.n * theta)

@dataclass
class OfflineFactors:
    """Container holding pre-computed matrices for fast online reconstruction."""
    W_inv: np.ndarray         # diag(1 / sqrt(ω_j)) (M, M)
    Phi_tilde: np.ndarray     # Φ·W_inv              (N, M)
    G: np.ndarray             # Φ~ Φ~ᵀ               (N, N)
    lam: float                # regularisation gain
    modes: List[Mode]         # basis (for field reconstruction)

class MocapServer:
    def __init__(self, config: MocapConfig = MocapConfig()):
        self.config = config
        self.mesh_marker_positions: Dict[int, List[float]] = {}
        self.rim_marker_positions: Dict[int, List[float]] = {marker_id: [0.0, 0.0, 0.0] for marker_id in [1, 2, 3]}
        self.normal = np.array([0, 0, -1])
        self.center_m = np.array([0, 0, 0])
        self.radius_m = 250.0*1e-3
        self.gap_m = 35.0*1e-3
        self.flag_calibrate = True
        self.plane_normal = np.array([0, 0, -1])
        self.plane_centroid = np.array([0, 0, 0])
        self.mesh_marker_pos = np.zeros((config.num_mesh_markers, 3))
        self.mesh_marker_pos_mm = np.zeros((config.num_mesh_markers, 3))
        self.mesh_marker_offset = np.zeros(config.num_mesh_markers)  # Store z-offsets
        self.offset_samples = []  # Store samples for averaging
        self.num_samples_needed = 10  # Number of samples to average
        self.display = DigitalDisplay(self, config.display_update_rate)
        self.last_update_time = time.time()  # Add timestamp for tracking updates
        self.voltage_data = None  # Store the latest voltage data
        self.last_coefficients = None  # Store the latest neural network coefficients
        if config.NN_enable:
            # Load the state_dict
            state_dict = torch.load(config.model_path, map_location=torch.device('cpu'), weights_only=True)  # Use 'cuda' if using GPU

            # Define Model & Training Parameters
            input_size = 3*config.num_mesh_markers+1  # Number of features in X_train
            hidden_size = config.hidden_size
            output_size = 3*config.n_basis  # Number of basis function coefficients
            # Initialize the neural network
            self.model = ShapeNet(input_size, hidden_size, output_size)

            # Load the state_dict into the model
            self.model.load_state_dict(state_dict)

            # Set device (CPU or GPU)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            self.model.eval()
            
    def __del__(self):
        """Cleanup when the server is destroyed."""
        if hasattr(self, 'display'):
            self.display.stop()

    def process_input(self, input):
        # Combine voltage data with mocap input
        voltage = [self.voltage_data] if self.voltage_data is not None else [0.0]
        voltage.extend(input)
        voltage = np.array(voltage, dtype=np.float32) # NN input must be float

        full_input = torch.from_numpy(voltage)
        
        with torch.no_grad():  # Disable gradient computation for evaluation
            predicted_coefficients = self.model(full_input).numpy().flatten()
            
        return predicted_coefficients

    def normalize_NN_input(self, x_mm, y_mm, z_mm, radius_m, gap_m):
        r_mm, theta = self.cartesian_to_polar_numpy(x_mm, y_mm)
        r_normalized = r_mm / (radius_m*1e3)
        z_normalized = z_mm / (gap_m*1e3) # covert z from meters to mm
        # Use zip and list comprehension to interleave
        normalized_input = [item for trio in zip(r_normalized, theta, z_normalized) for item in trio]
        return normalized_input

    def cartesian_to_polar_numpy(self, x, y):
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
        theta = np.arctan2(y, x)    # Angle in radians between -pi to pi
        
        return r, theta

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

    def initial_calibration(self) -> None:
        """Calculate rim parameters and initial z-offsets from marker positions."""
        rim_marker_pos = list(self.rim_marker_positions.values())

        # sort the markers by x position, note that this causes problems where markers switch IDs if they are close enough in x
        #mesh_marker_pos = sorted(list(self.mesh_marker_positions.values()), key=lambda pos: pos[0])
        mesh_marker_pos = list(self.mesh_marker_positions.values())
        
        # First calculate best fit plane from mesh markers
        if len(mesh_marker_pos) != self.config.num_mesh_markers:
            return

        # Convert positions to numpy array
        points = np.array(mesh_marker_pos)
        
        # Store sample for offset calculation
        if len(self.offset_samples) < self.num_samples_needed:
            self.offset_samples.append(points.copy())
            return  # Wait for more samples
        
        # Calculate average position from samples
        if len(self.offset_samples) == self.num_samples_needed:
            avg_positions = np.mean(self.offset_samples, axis=0)
            self.offset_samples = []  # Clear samples after calculating average
            
            # Calculate centroid from averaged positions
            self.plane_centroid = np.mean(avg_positions, axis=0)
            
            # Form the matrix A of mean-centered points
            A = avg_positions - self.plane_centroid
            
            # Calculate SVD
            _, _, vh = np.linalg.svd(A)
            
            # Normal vector is the last right singular vector
            normal = vh[-1]
            
            # Ensure normal points in negative z direction
            if normal[2] > 0:
                normal = -normal
            
            self.plane_normal = normal
            
            # Transform averaged positions to calculate offset
            transformed_pos = avg_positions - self.plane_centroid
            rotation_mat = self.rotation_matrix_from_vectors(self.plane_normal, self.config.z_axis)
            transformed_pos = transformed_pos @ rotation_mat.T
            
            # Store z-positions as offset (in millimeters)
            self.mesh_marker_offset = transformed_pos[:, 2] * 1000.0
            
            print("\nBest Fit Plane Calculation Results:")
            print(f"Plane Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
            print(f"Plane Centroid [mm]: [{self.plane_centroid[0]*1000:.1f}, {self.plane_centroid[1]*1000:.1f}, {self.plane_centroid[2]*1000:.1f}]")
            print("Initial z-offsets [mm]:", [f"{offset:.1f}" for offset in self.mesh_marker_offset])
            print("Finished Calculating Best Fit Plane\n")

            # Continue with rim calculation
            if not all(value != 0 for row in rim_marker_pos for value in row):
                return
            
            # Convert rim markers to numpy array
            rim_markers = np.array(rim_marker_pos)
            
            # Center the rim markers
            rim_markers = rim_markers - self.plane_centroid
            
            # Calculate rotation matrix using plane normal
            rotation_mat = self.rotation_matrix_from_vectors(self.plane_normal, self.config.z_axis)
            
            # Rotate rim markers
            rim_markers = rim_markers @ rotation_mat.T
            
            # Calculate circle parameters in rotated space
            p1, p2, p3 = rim_markers
            self.center_m, self.radius_m, self.normal = self.compute_circumradius(p1, p2, p3)
            
            # Convert center and radius to millimeters
            center_mm = self.center_m * 1000
            radius_mm = self.radius_m * 1000
            print("\nRim Calculation Results:")
            print(f"Center [mm]: [{center_mm[0]:.1f}, {center_mm[1]:.1f}, {center_mm[2]:.1f}]")
            print(f"Radius [mm]: {radius_mm:.1f}")
            print(f"Normal: [{self.normal[0]:.3f}, {self.normal[1]:.3f}, {self.normal[2]:.3f}]")
            self.flag_calibrate = False
            print("Finished Calculating Rim\n")

            # transform the marker positions
            self.transform_marker_positions()

            # calculate mocap probe points
            probe_x = self.mesh_marker_pos[:, 0]
            probe_y = self.mesh_marker_pos[:, 1]

            # ---------------------------- Build basis & measurement matrix -----------
            modes = self.build_modes(self.config.n_basis, self.config.m_basis)              # M modes
            Phi = self.build_measurement_matrix(probe_x, probe_y, modes)

            # ---------------------------- OFFLINE stage ------------------------------
            self.pre = self.precompute_factors(Phi, modes, lam=self.config.tikhonov_weight)

    def normalize_LS_input(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert (x, y) array → (r, θ)."""
        r = np.hypot(x, y) / self.radius_m
        theta = np.arctan2(y, x)
        return r, theta

    def build_measurement_matrix(self, probe_x: np.ndarray, probe_y: np.ndarray, modes: List[Mode]) -> np.ndarray:
        """Assemble Φ where Φ[i,j] = φ_j(r_i, θ_i)."""
        r, theta = self.normalize_LS_input(probe_x, probe_y)
        N = probe_x.shape[0]
        M = len(modes)
        Phi = np.empty((N, M), dtype=float)
        for j, mode in enumerate(modes):
            Phi[:, j] = mode.phi(r, theta)
        return Phi

    def recover_modes(self, z: np.ndarray, pre: OfflineFactors) -> np.ndarray:
        """Compute modal coefficients a from probe displacements z (length N)."""
        # Solve small N×N system: (G + λ² I) y = z
        N = pre.G.shape[0]
        rhs = np.linalg.solve(pre.G + pre.lam ** 2 * np.eye(N), z)
        # a = W_inv · Φ~ᵀ · y
        a = pre.W_inv @ (pre.Phi_tilde.T @ rhs)
        return a  # shape (M,)

    def build_modes(self, max_n: int = 3, max_m: int = 3) -> List[Mode]:
        """Generate Bessel modes up to n ≤ max_n and the first max_m radial roots.

        Returns a list of Mode objects.
        """
        modes: List[Mode] = []
        for n in range(max_n + 1):
            roots = jn_zeros(n, max_m)
            for k in roots:
                # cos term
                modes.append(Mode(n, k, "cos", omega=k**2))
                # add sin term for non-axisymmetric n≥1
                if n > 0:
                    modes.append(Mode(n, k, "sin", omega=k**2))
        return modes

    def precompute_factors(self, Phi: np.ndarray, modes: List[Mode], lam: float = 1e-3) -> OfflineFactors:
        """Compute and store matrices needed for the dual-form online solve."""
        omega = np.array([m.omega for m in modes])
        W_inv = np.diag(1.0 / np.sqrt(omega))            # (M, M)
        Phi_tilde = Phi @ W_inv                          # (N, M)
        G = Phi_tilde @ Phi_tilde.T                      # (N, N)
        return OfflineFactors(W_inv=W_inv, Phi_tilde=Phi_tilde, G=G, lam=lam, modes=modes)
    
    def transform_marker_positions(self) -> None:
        """Transform marker positions based on rim parameters and best fit plane."""
        #mesh_marker_pos = sorted(list(self.mesh_marker_positions.values()), key=lambda pos: pos[0])
        mesh_marker_pos = list(self.mesh_marker_positions.values())
        rim_marker_pos = list(self.rim_marker_positions.values())
        
        # Convert to numpy arrays
        mesh_marker_pos = np.array(mesh_marker_pos)
        rim_marker_pos = np.array(rim_marker_pos)

        # Transform points relative to hybrid center
        mesh_marker_pos = mesh_marker_pos - self.plane_centroid
        
        # Calculate rotation matrix using plane normal
        rotation_mat = self.rotation_matrix_from_vectors(self.plane_normal, self.config.z_axis)
        
        # Rotate points
        mesh_marker_pos = mesh_marker_pos @ rotation_mat.T
        # Translate x,y coordinates using center
        mesh_marker_pos[:, :2] = mesh_marker_pos[:, :2] - self.center_m[:2]
        
        # Store transformed positions in meters
        self.mesh_marker_pos = mesh_marker_pos
        
        # Convert to millimeters and apply offset if taring is enabled
        self.mesh_marker_pos_mm = mesh_marker_pos * 1000.0
        if self.config.tare:
            self.mesh_marker_pos_mm[:, 2] -= self.mesh_marker_offset
        
        self.display.update_positions(dict(zip(range(len(mesh_marker_pos)), self.mesh_marker_pos_mm)))

    def handle_client(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        """Handle interaction with a single connected client."""
        print(f"Connected by {addr}")
        conn.settimeout(0.1)
        TIMEOUT_THRESHOLD = 0.1  # 100 ms timeout threshold

        try:
            while True:
                try:
                    # Check for timeout
                    current_time = time.time()
                    if current_time - self.last_update_time > TIMEOUT_THRESHOLD:
                        print(f"Connection with {addr} timed out - no marker updates received for {TIMEOUT_THRESHOLD*1000:.0f}ms")
                        print("ERROR: Motion capture data timeout - exiting program")
                        conn.close()
                        sys.exit(1)  # Exit with error code 1

                    data = conn.recv(1024)
                    if not data:
                        print(f"Client {addr} closed the connection.")
                        break
                    
                    # Parse voltage values from received data
                    # Each voltage value is an 8-byte double
                    while len(data) >= 8:
                        voltage = struct.unpack('>d', data[:8])[0]  # Unpack 8 bytes as big-endian double
                        data = data[8:]  # Move to next 8 bytes
                    self.voltage_data = voltage  # Store the voltage data
                    
                    # Send all marker positions in millimeters
                    # Format: [x1,x2,x3,..., y1,y2,y3,..., z1,z2,z3,...]
                    x_mm = self.mesh_marker_pos_mm[:, 0]
                    y_mm = self.mesh_marker_pos_mm[:, 1]
                    z_mm = self.mesh_marker_pos_mm[:, 2]
                    flat_positions = np.concatenate([x_mm, y_mm, z_mm])
                    
                    # update the shape estimate
                    if self.config.NN_enable:
                        # normalize the input
                        mocap_input = self.normalize_NN_input(x_mm, y_mm, z_mm, self.radius_m, self.gap_m)
                        NN_predicted_coefficients = self.process_input(mocap_input)
                        self.last_coefficients = NN_predicted_coefficients
                        self.display.update_coefficients(NN_predicted_coefficients)

                    if self.config.LS_enable:
                        LS_predicted_coefficients = self.recover_modes(z_mm / (self.gap_m * 1e3), self.pre)

                        N_GRID = 50
                        r_full = np.linspace(0, 0.2, N_GRID)
                        theta_full = np.linspace(0, 2*np.pi, N_GRID)
                        Theta, R = np.meshgrid(theta_full, r_full, indexing='ij')
                        Z = self.reconstruct_field(LS_predicted_coefficients, self.pre, R, Theta)

                        nadir_index = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
                        nadir_r = R[nadir_index]
                        nadir_theta = Theta[nadir_index]
                        nadir_z_mm = np.max(Z) * self.gap_m * 1e3
                        nadir_x_mm = self.radius_m * 1e3 * nadir_r * np.cos(nadir_theta)
                        nadir_y_mm = self.radius_m * 1e3 * nadir_r * np.sin(nadir_theta)

                        self.last_coefficients = LS_predicted_coefficients
                        self.display.update_coefficients(LS_predicted_coefficients)
                        self.display.update_nadir(nadir_x_mm, nadir_y_mm, nadir_z_mm)

                    format_string = '>' + 'd' * len(flat_positions)
                    conn.sendall(struct.pack(format_string, *flat_positions))
                    
                except socket.timeout:
                    continue
                except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as e:
                    print(f"Connection error with {addr}: {e}")
                    break
        finally:
            conn.close()
            print(f"Connection with {addr} closed.\n\n\n")

    def reconstruct_field(self, a: np.ndarray, pre: OfflineFactors, grid_r: np.ndarray, grid_theta: np.ndarray) -> np.ndarray:
        """Return w(r, θ) on the provided polar grid."""
        w = np.zeros_like(grid_r)
        for coeff, mode in zip(a, pre.modes):
            w += coeff * mode.phi(grid_r, grid_theta)
        return w

    def receive_labeled_marker(self, marker_id: int, model_id: int, position: List[float]) -> None:
        """Handle labeled marker data from OptiTrack."""
        x, y, z = position
        if model_id == 0:  # mesh markers
            self.mesh_marker_positions[marker_id] = [y, z, x]
        elif model_id == 1:  # rim markers
            self.rim_marker_positions[marker_id] = [y, z, x]

        # Only proceed if we have all required markers
        if not (len(self.mesh_marker_positions) == self.config.num_mesh_markers):
            return

        # Calculate rim parameters once
        if self.flag_calibrate:
            self.initial_calibration()

        # Transform positions if calibration calculations are done
        if not self.flag_calibrate:
            self.transform_marker_positions()
            self.last_update_time = time.time()  # Update timestamp when positions are updated

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
        "enable_visualization": False,
        "enable_display": True,
        "tare": True,  # Default to true
        "z_scale": 1.0,  # Scale factor for z-direction visualization
        "NN_enable": False,  # Default to false
        "LS_enable": True,
    }
    
    if len(sys.argv) > 1:
        options["enable_visualization"] = sys.argv[1].lower() != "false"
    if len(sys.argv) > 2:
        options["enable_display"] = sys.argv[2].lower() != "false"
    if len(sys.argv) > 3:
        options["tare"] = sys.argv[3].lower() != "false"
    if len(sys.argv) > 4:
        options["z_scale"] = float(sys.argv[4])
    if len(sys.argv) > 5:
        options["NN_enable"] = sys.argv[5].lower() != "false"
    if len(sys.argv) > 6:
        options["LS_enable"] = sys.argv[6].lower() != "false"

    # Create config with tare option
    config = MocapConfig(tare=options["tare"], NN_enable=options["NN_enable"], LS_enable=options["LS_enable"])
    mocap_server = MocapServer(config)

    # Initialize NatNet client
    streaming_client = NatNetClient()
    streaming_client.set_nat_net_version(4, 1)
    streaming_client.set_client_address(options["clientAddress"])
    streaming_client.set_server_address(options["serverAddress"])
    streaming_client.set_use_multicast(options["use_multicast"])
    streaming_client.set_print_level(0)

    streaming_client.labeled_marker_listener = mocap_server.receive_labeled_marker

    # Start the NatNet client
    if not streaming_client.run():
        print("ERROR: Could not start streaming client.")
        sys.exit(1)
    
    # Initialize and start the visualizer if enabled
    visualizer = None
    if options["enable_visualization"]:
        from mocap_visualizer import MoCapVisualizer
        visualizer = MoCapVisualizer(mocap_server, z_scale=options["z_scale"])
        visualizer.start()
        print(f"3D visualization started with z-scale: {options['z_scale']}")

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