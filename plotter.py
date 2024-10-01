import sys
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QFont, QPainter
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

class LabeledScatterPlot:
    def __init__(self, num_points=100):
        # Initialize the Qt Application
        self.app = QApplication(sys.argv)
        
        # Set up the OpenGL view widget
        self.view = gl.GLViewWidget()
        self.view.show()
        self.view.setWindowTitle('Real-time 3D Scatter Plot with Labels')
        self.view.setCameraPosition(distance=40)
        
        # Initialize scatter plot data
        self.num_points = num_points
        self.positions = np.random.uniform(-10, 10, size=(self.num_points, 3))
        
        # Create the scatter plot item
        self.scatter = gl.GLScatterPlotItem(pos=self.positions, size=5, color=(1, 1, 1, 1), pxMode=False)
        self.view.addItem(self.scatter)
        
        # Set up a QGraphicsView for labels
        self.label_view = QGraphicsView()
        self.label_view.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.label_view.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.label_view.setAttribute(Qt.WA_NoSystemBackground)
        self.label_view.setAttribute(Qt.WA_TranslucentBackground)
        self.label_view.setScene(QGraphicsScene())
        self.label_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.label_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.label_view.showFullScreen()
        
        # Initialize labels
        self.labels = []
        font = QFont("Arial", 10)
        for i in range(self.num_points):
            label = self.label_view.scene().addText(str(i), font)
            label.setDefaultTextColor(Qt.red)
            self.labels.append(label)
        
        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(5)  # Update every 5 ms
        
        # Connect the view's render signal to update labels after rendering
        self.view.camera().linkView(self.view)
        self.view.renderingOrderChanged.connect(self.update_labels)
    
    def update(self):
        # Update the positions with new data
        self.positions = np.random.uniform(-10, 10, size=(self.num_points, 3))
        self.scatter.setData(pos=self.positions)
        self.update_labels()
    
    def update_labels(self):
        # Project 3D positions to 2D screen coordinates
        view_matrix = self.view.cameraParams()
        projection_matrix = self.view.projectionMatrix()
        width = self.view.width()
        height = self.view.height()
        
        for i, pos in enumerate(self.positions):
            # Convert 3D to 2D
            screen_pos = self.view.camera().project(pos)
            if screen_pos is not None:
                x, y = screen_pos
                # Position the label
                self.labels[i].setPos(x, y)
                self.labels[i].show()
            else:
                self.labels[i].hide()
        
        # Update the label view
        self.label_view.viewport().update()
    
    def run(self):
        sys.exit(self.app.exec_())

if __name__ == '__main__':
    plot = LabeledScatterPlot(num_points=100)
    plot.run()
