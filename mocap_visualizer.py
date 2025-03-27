import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend for interactive plotting

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
        self.thread = None
        
        # Initialize the figure and axes
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.mesh_scatter = None
        self.rim_scatter = None
        self.circle_line = None
        
        # Initialize data structures
        self.mesh_marker_pos_vis = np.zeros((mocap_server.config.num_mesh_markers, 3))
        self.rim_marker_pos_vis = np.zeros((3, 3))
        self.circle_points = np.zeros((100, 3))
        
        # Lock for thread-safe data access
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
    
    def init_plot(self):
        """Initialize the 3D plot elements"""
        self.ax.set_xlabel('Y')
        self.ax.set_ylabel('Z')
        self.ax.set_zlabel('X')
        self.ax.set_title('Motion Capture Visualization')
        
        # Create scatter plots for mesh and rim markers
        self.mesh_scatter = self.ax.scatter([], [], [], c='b', marker='o', s=50, label='Mesh Markers')
        self.rim_scatter = self.ax.scatter([], [], [], c='r', marker='s', s=100, label='Rim Markers')
        
        # Create line for circle
        self.circle_line, = self.ax.plot([], [], [], 'g-', linewidth=2, label='Calculated Rim')
        
        # Add legend
        self.ax.legend()
        
        # Set initial view limits
        self.ax.set_xlim([-0.5, 0.5])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_zlim([-0.5, 0.5])
        
        return self.mesh_scatter, self.rim_scatter, self.circle_line
    
    def update_plot(self, frame):
        """Update function for animation"""
        self.update_visualization_data()
        
        with self.data_lock:
            # Update mesh markers
            if len(self.mesh_marker_pos_vis) > 0:
                self.mesh_scatter._offsets3d = (
                    self.mesh_marker_pos_vis[:, 0],
                    self.mesh_marker_pos_vis[:, 1],
                    self.mesh_marker_pos_vis[:, 2]
                )
            
            # Update rim markers
            self.rim_scatter._offsets3d = (
                self.rim_marker_pos_vis[:, 0],
                self.rim_marker_pos_vis[:, 1],
                self.rim_marker_pos_vis[:, 2]
            )
            
            # Update circle
            self.circle_line.set_data(self.circle_points[:, 0], self.circle_points[:, 1])
            self.circle_line.set_3d_properties(self.circle_points[:, 2])
            
            # Auto-adjust axes limits if we have data
            if np.any(self.mesh_marker_pos_vis) or np.any(self.rim_marker_pos_vis):
                all_points = np.vstack([self.mesh_marker_pos_vis, self.rim_marker_pos_vis, self.circle_points])
                max_range = np.max([
                    np.max(all_points[:, 0]) - np.min(all_points[:, 0]),
                    np.max(all_points[:, 1]) - np.min(all_points[:, 1]),
                    np.max(all_points[:, 2]) - np.min(all_points[:, 2])
                ])
                mid_x = (np.max(all_points[:, 0]) + np.min(all_points[:, 0])) / 2
                mid_y = (np.max(all_points[:, 1]) + np.min(all_points[:, 1])) / 2
                mid_z = (np.max(all_points[:, 2]) + np.min(all_points[:, 2])) / 2
                
                self.ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
                self.ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
                self.ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        return self.mesh_scatter, self.rim_scatter, self.circle_line
    
    def visualization_loop(self):
        """Main loop for the visualization thread"""
        self.init_plot()
        ani = FuncAnimation(
            self.fig, self.update_plot, 
            frames=None, 
            init_func=self.init_plot,
            interval=1000/self.update_rate,  # interval in milliseconds
            blit=True
        )
        plt.show()
        
    def start(self):
        """Start the visualization thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.visualization_loop)
            self.thread.daemon = True  # Thread will exit when main program exits
            self.thread.start()
            print("Visualization thread started")
    
    def stop(self):
        """Stop the visualization thread"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=1.0)
                print("Visualization thread stopped")

# Example usage:
# visualizer = MocapVisualizer(mocap_server)
# visualizer.start()
# ...
# visualizer.stop()  # When shutting down