import numpy as np
import time
import multiprocessing
from multiprocessing import Process, Queue
import threading
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

class MoCapVisualizer:
    def __init__(self, mocap_server, update_rate=30):
        """
        Initialize the visualizer with a reference to the mocap server
        
        Args:
            mocap_server: The MocapServer instance to visualize
            update_rate: Visualization update rate in Hz (default: 30)
        """
        self.mocap_server = mocap_server
        self.update_rate = update_rate
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
            # Deep copy the data to avoid issues with concurrent access
            if self.mocap_server.mesh_marker_positions:
                mesh_marker_pos = list(self.mocap_server.mesh_marker_positions.values())
                if len(mesh_marker_pos) == self.mocap_server.config.num_mesh_markers:
                    # Apply the same transformations as in mocap_forwarding
                    mesh_marker_pos = np.array(mesh_marker_pos)
                    # First center on plane centroid
                    mesh_marker_pos = mesh_marker_pos - self.mocap_server.plane_centroid
                    # Rotate to align with z-axis using plane normal
                    rotation_mat = self.mocap_server.rotation_matrix_from_vectors(
                        self.mocap_server.plane_normal, 
                        self.mocap_server.config.z_axis
                    )
                    mesh_marker_pos = mesh_marker_pos @ rotation_mat.T
                    # Translate x,y coordinates using rim center
                    mesh_marker_pos[:, :2] = mesh_marker_pos[:, :2] - self.mocap_server.center[:2]
                    self.mesh_marker_pos_vis = mesh_marker_pos
            
            rim_marker_pos = list(self.mocap_server.rim_marker_positions.values())
            if all(value != 0 for row in rim_marker_pos for value in row):
                # Apply the same transformations to rim markers
                rim_marker_pos = np.array(rim_marker_pos)
                rim_marker_pos = rim_marker_pos - self.mocap_server.plane_centroid
                rim_marker_pos = rim_marker_pos @ rotation_mat.T
                rim_marker_pos[:, :2] = rim_marker_pos[:, :2] - self.mocap_server.center[:2]
                self.rim_marker_pos_vis = rim_marker_pos
                
                # Calculate the circle points in the transformed space
                center = np.array([0, 0, self.mocap_server.center[2]])  # Center should be at x,y=0 in transformed space
                normal = np.array([0, 0, -1])  # Normal should be aligned with z-axis
                
                if not np.array_equal(normal, np.array([0, 0, 0])):
                    # Create a circle in the XY plane
                    theta = np.linspace(0, 2*np.pi, 100)
                    radius = self.mocap_server.radius
                    
                    # Generate circle points directly in XY plane
                    circle_points = np.zeros((100, 3))
                    circle_points[:, 0] = radius * np.cos(theta)
                    circle_points[:, 1] = radius * np.sin(theta)
                    circle_points[:, 2] = center[2]
                    
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
    
    Args:
        data_queue: Queue for receiving visualization data
        config_num_mesh_markers: Number of mesh markers
        update_rate: Update rate in Hz
    """
    # Enable anti-aliasing for prettier plots
    pg.setConfigOptions(antialias=True)
    
    # Create the application
    app = pg.mkQApp("MoCap Visualization")
    
    # Create the 3D visualization window
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
    
    # Create empty scatter plot items for mesh and rim markers
    mesh_scatter = gl.GLScatterPlotItem(
        pos=np.zeros((config_num_mesh_markers, 3)),
        color=(0, 0, 1, 1),
        size=10,
        pxMode=True
    )
    window.addItem(mesh_scatter)
    
    rim_scatter = gl.GLScatterPlotItem(
        pos=np.zeros((3, 3)),
        color=(1, 0, 0, 1),
        size=15,
        pxMode=True
    )
    window.addItem(rim_scatter)
    
    # Create line for circle
    circle_line = gl.GLLinePlotItem(
        pos=np.zeros((100, 3)),
        color=(0, 1, 0, 1),
        width=2,
        mode='line_strip'
    )
    window.addItem(circle_line)
    
    # Create a timer for updates
    timer = QtCore.QTimer()
    
    # Set up update function
    def update():
        # Check for new data
        try:
            while not data_queue.empty():
                data = data_queue.get_nowait()
                mesh_pos = data['mesh']
                rim_pos = data['rim']
                circle_points = data['circle']
                
                # Update the visualizations
                if len(mesh_pos) > 0:
                    mesh_scatter.setData(pos=mesh_pos)
                
                rim_scatter.setData(pos=rim_pos)
                circle_line.setData(pos=circle_points)
                
                # Auto-adjust camera if needed
                # (PyQtGraph's auto range in 3D is not as straightforward as in Matplotlib)
        except Exception as e:
            print(f"Error in visualization update: {e}")
    
    # Connect timer to update function
    timer.timeout.connect(update)
    timer.start(int(1000 / update_rate))  # interval in milliseconds
    
    # Start the Qt application event loop
    pg.exec()


# Example usage:
# visualizer = HighPerformanceVisualizer(mocap_server)
# visualizer.start()
# ...
# visualizer.stop()  # When shutting down

if __name__ == "__main__":
    # This allows the file to be run directly for testing
    print("This module is meant to be imported, not run directly.")