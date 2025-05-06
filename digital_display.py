import os
import time
from threading import Thread, Event
from typing import Dict, List

class DigitalDisplay:
     def __init__(self, mocap_server, update_rate: float = 10.0):
         self.mocap_server = mocap_server
         self.update_rate = update_rate
         self.stop_event = Event()
         self.display_thread = None
         self.last_positions = None
         self.last_coefficients = None
 
     def clear_screen(self):
         os.system('cls' if os.name == 'nt' else 'clear')
 
     def format_value(self, value: float) -> str:
         return f"{value:6.2f}"
 
     def display_positions(self):
         """Display the current mesh marker positions in a formatted table."""
         self.clear_screen()
         print("\n=== Mesh Marker Positions (mm) ===")
         print("Marker ID |    X    |    Y    |    Z    | Offset  |")
         print("-" * 49)
         
         if self.last_positions is None:
             print("Waiting for marker data...")
             return
 
         for marker_id, pos in sorted(self.last_positions.items()):
             offset = self.mocap_server.mesh_marker_offset[marker_id]
             print(f"{marker_id:8d} | {self.format_value(pos[0])} | {self.format_value(pos[1])} | {self.format_value(pos[2])} | {self.format_value(offset)} |")
         
         print("-" * 49)
         print(f"Last update: {time.strftime('%H:%M:%S')}")
         if self.mocap_server.config.tare:
             print("Tare: ON")
         else:
             print("Tare: OFF")

         # Display neural network coefficients if enabled
         if self.mocap_server.config.NN_enable and self.last_coefficients is not None:
             print("\n=== Neural Network Coefficients ===")
             for i, coef in enumerate(self.last_coefficients):
                 print(f"Coefficient {i+1}: {self.format_value(coef)}")
         
         print("\nPress Ctrl+C to exit")
 
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

     def update_coefficients(self, coefficients: List[float]):
         """Update the stored neural network coefficients."""
         self.last_coefficients = coefficients