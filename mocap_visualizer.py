import threading
import time
import numpy as np
import multiprocessing
from multiprocessing import Process, Queue
import matplotlib
# Force matplotlib to use a specific backend before any other imports
matplotlib.use('Qt5Agg')  # Use Qt5 backend for interactive plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def visualization_process(data_queue, config_num_mesh_markers, update_rate=30):
    """
    Separate process function for visualization
    
    Args:
        data_queue: Queue for receiving visualization data
        config_num_mesh_markers: Number of mesh markers
        update_rate: Update rate in Hz
    """
    # Initialize the figure and axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set labels and title
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_zlabel('X')
    ax.set_title('Motion Capture Visualization')
    
    # Initialize data structures
    mesh_marker_pos_vis = np.zeros((config_num_mesh_markers, 3))
    rim_marker_pos_vis = np.zeros((3, 3))
    circle_points = np.zeros((100, 3))
    
    # Create scatter plots for mesh and rim markers
    mesh_scatter = ax.scatter([], [], [], c='b', marker='o', s=50, label='Mesh Markers')
    rim_scatter = ax.scatter([], [], [], c='r', marker='s', s=100, label='Rim Markers')
    
    # Create line for circle
    circle_line, = ax.plot([], [], [], 'g-', linewidth=2, label='Calculated Rim')
    
    # Set initial view limits
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    
    # Add legend
    ax.legend()
    
    def update_plot(frame):
        nonlocal mesh_marker_pos_vis, rim_marker_pos_vis, circle_points
        
        # Check for new data
        try:
            while not data_queue.empty():
                data = data_queue.get_nowait()
                mesh_marker_pos_vis = data['mesh']
                rim_marker_pos_vis = data['rim']
                circle_points = data['circle']
        except Exception as e:
            print(f"Error getting data from queue: {e}")
        
        # Update mesh markers
        if len(mesh_marker_pos_vis) > 0:
            mesh_scatter._offsets3d = (
                mesh_marker_pos_vis[:, 0],
                mesh_marker_pos_vis[:, 1],
                mesh_marker_pos_vis[:, 2]
            )
        
        # Update rim markers
        rim_scatter._offsets3d = (
            rim_marker_pos_vis[:, 0],
            rim_marker_pos_vis[:, 1],
            rim_marker_pos_vis[:, 2]
        )
        
        # Update circle
        circle_line.set_data(circle_points[:, 0], circle_points[:, 1])
        circle_line.set_3d_properties(circle_points[:, 2])
        
        # Auto-adjust axes limits if we have data
        if np.any(mesh_marker_pos_vis) or np.any(rim_marker_pos_vis):
            all_points = np.vstack([mesh_marker_pos_vis, rim_marker_pos_vis, circle_points])
            max_range = np.max([
                np.max(all_points[:, 0]) - np.min(all_points[:, 0]),
                np.max(all_points[:, 1]) - np.min(all_points[:, 1]),
                np.max(all_points[:, 2]) - np.min(all_points[:, 2])
            ])
            mid_x = (np.max(all_points[:, 0]) + np.min(all_points[:, 0])) / 2
            mid_y = (np.max(all_points[:, 1]) + np.min(all_points[:, 1])) / 2
            mid_z = (np.max(all_points[:, 2]) + np.min(all_points[:, 2])) / 2
            
            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        return mesh_scatter, rim_scatter, circle_line
    
    # Create the animation
    ani = FuncAnimation(
        fig, update_plot, 
        frames=None,
        interval=1000/update_rate,
        blit=True
    )
    
    # Show the plot (this will block until the window is closed)
    plt.show()


class MocapVisualizer:
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
                    self.mesh_marker_pos_vis = np.array(mesh_marker_pos)
            
            rim_marker_pos = list(self.mocap_server.rim_marker_positions.values())
            if all(value != 0 for row in rim_marker_pos for value in row):
                self.rim_marker_pos_vis = np.array(rim_marker_pos)
                
                # Calculate the circle points for visualization
                center = self.mocap_server.center
                normal = self.mocap_server.normal
                
                if not np.array_equal(normal, np.array([0, 0, 0])):
                    # Create a circle in the plane defined by the normal
                    # First, find a vector perpendicular to the normal
                    if np.abs(normal[0]) < np.abs(normal[1]):
                        v1 = np.array([1, 0, 0])
                    else:
                        v1 = np.array([0, 1, 0])
                    
                    v1 = v1 - normal * np.dot(v1, normal)
                    v1 = v1 / np.linalg.norm(v1)
                    
                    # Find a second perpendicular vector
                    v2 = np.cross(normal, v1)
                    v2 = v2 / np.linalg.norm(v2)
                    
                    # Calculate the radius as the distance from center to any rim marker
                    radius = np.linalg.norm(self.rim_marker_pos_vis[0] - center)
                    
                    # Generate circle points
                    theta = np.linspace(0, 2*np.pi, 100)
                    circle_points = np.zeros((100, 3))
                    for i, angle in enumerate(theta):
                        circle_points[i] = center + radius * (np.cos(angle) * v1 + np.sin(angle) * v2)
                    
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
            
            print("Visualization started in separate process")
    
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

# Example usage:
# visualizer = MocapVisualizer(mocap_server)
# visualizer.start()
# ...
# visualizer.stop()  # When shutting down

if __name__ == "__main__":
    # This allows the file to be run directly for testing
    print("This module is meant to be imported, not run directly.")