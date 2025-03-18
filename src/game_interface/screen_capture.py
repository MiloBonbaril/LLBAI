import numpy as np
import cv2
import mss
import time
from typing import Tuple, Optional, Dict, Any


class ScreenCapture:
    """
    Captures gameplay frames from Lethal League Blaze window.
    """
    
    def __init__(self, window_name: str = "LLBlaze"):
        """
        Initialize the screen capture module.
        
        Args:
            window_name: Title of the game window to capture.
        """
        self.window_name = window_name
        self.sct = mss.mss()
        self.monitor = None
        self.last_frame_time = 0
        self.fps_limit = 60
        self.frame_interval = 1.0 / self.fps_limit
        
    def find_game_window(self) -> bool:
        """
        Find the game window by its title.
        
        Returns:
            bool: True if window was found, False otherwise.
        """
        try:
            # This implementation depends on the OS
            # Windows implementation with win32gui
            import win32gui
            
            def callback(hwnd, extra):
                if win32gui.IsWindowVisible(hwnd) and self.window_name in win32gui.GetWindowText(hwnd):
                    rect = win32gui.GetWindowRect(hwnd)
                    x, y, width, height = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
                    
                    # Add a small offset to account for window borders and title bar
                    self.monitor = {
                        "left": x + 8,  # Left border offset
                        "top": y + 32,  # Title bar offset
                        "width": width - 16,  # Width minus borders
                        "height": height - 40,  # Height minus borders and title bar
                        "mon": 0  # Required by mss
                    }
                    return True
            
            win32gui.EnumWindows(callback, None)
            return self.monitor is not None
            
        except ImportError:
            print("win32gui not installed. Please install with: pip install pywin32")
            # Fallback to full screen capture if window-specific capture isn't available
            monitors = self.sct.monitors
            self.monitor = monitors[1]  # First non-primary monitor, or use monitors[0] for primary
            return True
    
    def capture_screen(self, resize: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Capture the current game screen.
        
        Args:
            resize: Optional (width, height) to resize the captured image.
            
        Returns:
            np.ndarray: The captured screen as a numpy array (BGR format).
        """
        # Throttle capture rate to maintain consistent FPS
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        
        if time_diff < self.frame_interval:
            time.sleep(self.frame_interval - time_diff)
        
        # Find window if not already found
        if self.monitor is None:
            success = self.find_game_window()
            if not success:
                raise RuntimeError(f"Could not find game window: {self.window_name}")
        
        # Capture screen
        sct_img = self.sct.grab(self.monitor)
        
        # Convert to numpy array
        img = np.array(sct_img)
        
        # Convert BGRA to BGR (remove alpha channel)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Resize if specified
        if resize:
            img = cv2.resize(img, resize)
        
        self.last_frame_time = time.time()
        return img
    
    def set_fps_limit(self, fps: int) -> None:
        """
        Set the maximum capture frame rate.
        
        Args:
            fps: Frames per second to limit capture to.
        """
        self.fps_limit = fps
        self.frame_interval = 1.0 / self.fps_limit
    
    def get_resolution(self) -> Tuple[int, int]:
        """
        Get the current capture resolution.
        
        Returns:
            Tuple[int, int]: (width, height) of the captured screen.
        """
        if self.monitor is None:
            success = self.find_game_window()
            if not success:
                raise RuntimeError(f"Could not find game window: {self.window_name}")
                
        return self.monitor["width"], self.monitor["height"]


# For testing purposes
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create screen capture
    capture = ScreenCapture()
    
    # Test capture
    print("Capturing screen...")
    frame = capture.capture_screen()
    
    # Display captured frame
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Captured Game Screen")
    plt.axis('off')
    plt.show()
    
    print(f"Captured frame shape: {frame.shape}")
    print(f"Game window resolution: {capture.get_resolution()}")