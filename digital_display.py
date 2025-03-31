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
        self.last_update_time = 0
        self.last_display_lines = 0
        
        # Get terminal size
        self.terminal_width = os.get_terminal_size().columns
        self.terminal_height = os.get_terminal_size().lines
        
        # Initialize display buffer
        self.display_buffer = []
        
        # Clear screen once at initialization
        self._clear_screen()

    def _clear_screen(self):
        """Clear the screen using ANSI escape codes."""
        print('\033[2J\033[H', end='')  # Clear screen and move cursor to home position

    def _get_display_lines(self):
        """Calculate the number of lines needed for the display."""
        if self.last_positions is None:
            return 5  # Header + "Waiting for data" + separator + timestamp + prompt
        return 5 + len(self.last_positions)  # Header + data lines + separator + timestamp + prompt

    def _create_display_buffer(self):
        """Create the display buffer with current data."""
        buffer = []
        
        # Header
        buffer.append("\n=== Mesh Marker Positions (mm) ===")
        buffer.append("Marker ID |    X    |    Y    |    Z    |")
        buffer.append("-" * 40)
        
        if self.last_positions is None:
            buffer.append("Waiting for marker data...")
        else:
            for marker_id, pos in sorted(self.last_positions.items()):
                buffer.append(f"{marker_id:8d} | {self.format_value(pos[0])} | {self.format_value(pos[1])} | {self.format_value(pos[2])} |")
        
        buffer.append("-" * 40)
        buffer.append(f"Last update: {time.strftime('%H:%M:%S')}")
        buffer.append("\nPress Ctrl+C to exit")
        
        return buffer

    def _update_display(self):
        """Update the display with new data."""
        # Check if terminal size has changed
        current_width = os.get_terminal_size().columns
        current_height = os.get_terminal_size().lines
        if current_width != self.terminal_width or current_height != self.terminal_height:
            self.terminal_width = current_width
            self.terminal_height = current_height
            self._clear_screen()
        
        # Create new display buffer
        new_buffer = self._create_display_buffer()
        new_lines = len(new_buffer)
        
        # If we have a previous display, move cursor up to overwrite it
        if self.last_display_lines > 0:
            print(f"\033[{self.last_display_lines}A", end='')
        
        # Print new display
        print('\n'.join(new_buffer))
        
        # Store the number of lines we just printed
        self.last_display_lines = new_lines

    def format_value(self, value: float) -> str:
        return f"{value:6.2f}"

    def update_loop(self):
        """Main update loop for the display."""
        while not self.stop_event.is_set():
            current_time = time.time()
            if current_time - self.last_update_time >= 1.0 / self.update_rate:
                self._update_display()
                self.last_update_time = current_time
            time.sleep(0.01)  # Small sleep to prevent CPU overuse

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
        # Clear the screen one last time
        self._clear_screen()

    def update_positions(self, positions: Dict[int, List[float]]):
        """Update the stored positions."""
        self.last_positions = positions 