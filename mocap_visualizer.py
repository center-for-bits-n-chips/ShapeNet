import numpy as np
import time
import multiprocessing
from multiprocessing import Process, Queue
import threading
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

class MoCapVisualizer:
    def __init__(self, mocap_server, update_rate=30, z_scale=1.0):
        """
        Initialize the visualizer with a reference to the mocap server
        
        Args:
            mocap_server: The MocapServer instance to visualize
            update_rate: Visualization update rate in Hz (default: 30)
            z_scale: Scale factor for z-direction visualization (default: 1.0)
        """
        self.mocap_server = mocap_server
        self.update_rate = update_rate
        self.z_scale = z_scale
        self.running = False
        self.process = None
        self.data_queue = None
        
        # Initialize data structures
        self.mesh_marker_pos_vis = np.zeros((mocap_server.config.num_mesh_markers, 3))
        self.rim_marker_pos_vis = np.zeros((3, 3))
        self.circle_points = np.zeros((100, 3))
        
        # Thread for data collection
        self.collector_thread = None
        self.data_lock = threading.Lock()
    
    def update_visualization_data(self):
        """Update the visualization data from the mocap server in a thread-safe manner"""
        with self.data_lock:
            # Use the already transformed mesh marker positions and apply z-scaling
            self.mesh_marker_pos_vis = self.mocap_server.mesh_marker_pos.copy()
            self.mesh_marker_pos_vis[:, 2] *= self.z_scale
            
            # Create circle in XY plane (already in transformed space)
            theta = np.linspace(0, 2*np.pi, 100)
            radius = self.mocap_server.radius
            
            # Generate circle points
            circle_points = np.zeros((100, 3))
            circle_points[:, 0] = radius * np.cos(theta)
            circle_points[:, 1] = radius * np.sin(theta)
            circle_points[:, 2] = 0.0
            
            self.circle_points = circle_points
    
    def data_collection_thread(self):
        """Thread function to collect data and send to visualization process"""
        while self.running:
            self.update_visualization_data()
            
            # Package data to send to the visualization process
            with self.data_lock:
                data_package = {
                    'mesh': self.mesh_marker_pos_vis.copy(),
                    'rim': self.rim_marker_pos_vis.copy(),
                    'circle': self.circle_points.copy()
                }
                
                # Add neural network coefficients if available
                if self.mocap_server.config.NN_enable and self.mocap_server.last_coefficients is not None:
                    data_package['coefficients'] = self.mocap_server.last_coefficients
            
            # Send data to visualization process if queue isn't full
            try:
                if not self.data_queue.full():
                    self.data_queue.put_nowait(data_package)
            except:
                pass
            
            # Sleep to maintain update rate
            time.sleep(1.0 / self.update_rate)

    def start(self):
        """Start the visualization process and data collection thread"""
        if not self.running:
            self.running = True
            
            # Create a queue for communication between the main process and visualization process
            self.data_queue = multiprocessing.Queue(maxsize=10)  # Limit queue size to prevent memory issues
            
            # Start the visualization process
            self.process = Process(
                target=visualization_process,
                args=(self.data_queue, self.mocap_server.config.num_mesh_markers, self.update_rate)
            )
            self.process.daemon = True  # Process will exit when main program exits
            self.process.start()
            
            # Start the data collection thread
            self.collector_thread = threading.Thread(target=self.data_collection_thread)
            self.collector_thread.daemon = True
            self.collector_thread.start()
            
            print("High-performance visualization started in separate process")
    
    def stop(self):
        """Stop the visualization process and data collection thread"""
        if self.running:
            self.running = False
            
            # Stop the data collection thread
            if self.collector_thread:
                self.collector_thread.join(timeout=1.0)
            
            # Close the visualization process
            if self.process and self.process.is_alive():
                # Give it a moment to clean up
                time.sleep(0.5)
                self.process.terminate()
                self.process.join(timeout=1.0)
            
            # Clean up the queue
            if self.data_queue:
                while not self.data_queue.empty():
                    try:
                        self.data_queue.get_nowait()
                    except:
                        pass
            
            print("Visualization stopped")


def visualization_process(data_queue, config_num_mesh_markers, update_rate=30):
    """
    Separate process function for visualization using PyQtGraph
    """
    pg.setConfigOptions(antialias=True)
    app = pg.mkQApp("MoCap Visualization")
    
    window = gl.GLViewWidget()
    window.setWindowTitle('MoCap Visualization')
    window.setCameraPosition(distance=2, elevation=30, azimuth=45)
    window.show()
    
    # Add a grid
    grid = gl.GLGridItem()
    grid.setSize(2, 2)
    grid.setSpacing(0.1, 0.1)
    window.addItem(grid)
    
    # Add axes
    axis = gl.GLAxisItem()
    axis.setSize(0.5, 0.5, 0.5)
    window.addItem(axis)
    
    # Create empty scatter plot items for mesh markers
    mesh_scatter = gl.GLScatterPlotItem(
        pos=np.zeros((config_num_mesh_markers, 3)),
        color=(0, 0, 1, 1),
        size=10,
        pxMode=True
    )
    window.addItem(mesh_scatter)
    
    # Create line for circle
    circle_line = gl.GLLinePlotItem(
        pos=np.zeros((100, 3)),
        color=(0, 1, 0, 1),
        width=2,
        mode='line_strip'
    )
    window.addItem(circle_line)

    # Create initial mesh for ShapeNet visualization
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
    meshdata = gl.MeshData(vertexes=init_vertices, faces=init_faces)
    
    # Create the mesh item
    shape_mesh = gl.GLMeshItem(
        meshdata=meshdata,
        smooth=False,
        color=(0.5, 0.5, 1, 1),  # RGBA
        shader='shaded',
        drawEdges=True,
    )
    window.addItem(shape_mesh)
    
    # Create a timer for updates
    timer = QtCore.QTimer()
    
    def update():
        try:
            while not data_queue.empty():
                data = data_queue.get_nowait()
                mesh_pos = data['mesh']
                circle_points = data['circle']
                
                if len(mesh_pos) > 0:
                    mesh_scatter.setData(pos=mesh_pos)
                circle_line.setData(pos=circle_points)

                # Update ShapeNet mesh if coefficients are available
                if 'coefficients' in data and data['coefficients'] is not None:
                    # Generate mesh grid for reconstruction
                    r_full = np.linspace(0, 1.0, 50)
                    theta_full = np.linspace(0, 2*np.pi, 50)
                    Theta, R = np.meshgrid(theta_full, r_full, indexing='ij')

                    # Generate basis functions
                    basis_functions = generate_basis_functions_for_surface(R, Theta, 3)  # Using 3 basis functions
                    Z = np.dot(basis_functions, data['coefficients'])
                    
                    # Convert to Cartesian coordinates
                    X = R * np.cos(Theta)
                    Y = R * np.sin(Theta)

                    # Scale for visualization using mocap server's z_scale
                    Z *= 1.0

                    # Create color gradient based on Z values
                    colors = np.zeros((X.size, 4), dtype=float)
                    colors[:, 0] = (Z.flatten() - Z.min()) / (Z.max() - Z.min())  # Red channel
                    colors[:, 1] = 0.5  # Green channel
                    colors[:, 2] = 1 - (Z.flatten() - Z.min()) / (Z.max() - Z.min())  # Blue channel
                    colors[:, 3] = 1.0  # Alpha channel

                    # Create vertices and faces
                    vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
                    faces = create_faces(X, Y)

                    # Update mesh data
                    meshdata = gl.MeshData(vertexes=vertices, faces=faces)
                    meshdata.setVertexColors(colors)
                    shape_mesh.setMeshData(meshdata=meshdata)

        except Exception as e:
            print(f"Error in visualization update: {e}")
    
    timer.timeout.connect(update)
    timer.start(int(1000 / update_rate))
    
    pg.exec()

def create_faces(X, Y):
    """Create a list of triangle faces from meshgrid coordinates."""
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

def generate_basis_functions_for_surface(r, theta, n_basis):
    """Generate basis functions for surface reconstruction."""
    from scipy.special import jn_zeros, jv
    
    # Create Bessel Function Zeros Table
    alphas = []
    for m in range(2):
        zeros = jn_zeros(m, n_basis)
        alphas.extend(zeros)
    
    basis_functions = []
    for k in range(n_basis):
        n = k + 1  # indexing by n = 1, 2, 3
        phi_n = jv(0, alphas[k]*r)
        phi_n_sin = jv(1, alphas[n_basis + k]*r) * np.sin(theta)
        phi_n_cos = jv(1, alphas[n_basis + k]*r) * np.cos(theta)
        basis_functions.append(phi_n)
        basis_functions.append(phi_n_sin)
        basis_functions.append(phi_n_cos)
    
    return np.stack(basis_functions, axis=2)  # Shape: (n_samples, n_basis)


# Example usage:
# visualizer = HighPerformanceVisualizer(mocap_server)
# visualizer.start()
# ...
# visualizer.stop()  # When shutting down

if __name__ == "__main__":
    # This allows the file to be run directly for testing
    print("This module is meant to be imported, not run directly.")