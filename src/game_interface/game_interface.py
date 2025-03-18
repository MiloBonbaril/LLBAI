import time
import os
from typing import Tuple, Dict, Any, Optional, List
import numpy as np

# Import local modules
from game_interface.screen_capture import ScreenCapture
from game_interface.input_controller import InputController


class GameInterface:
    """
    Main interface for interacting with Lethal League Blaze.
    Combines screen capture and input control.
    """
    
    def __init__(self, 
                 window_name: str = "Lethal League Blaze",
                 fps_limit: int = 30):
        """
        Initialize the game interface.
        
        Args:
            window_name: Title of the game window.
            fps_limit: Maximum frame rate for screen capture.
        """
        self.screen_capture = ScreenCapture(window_name)
        self.input_controller = InputController()
        self.fps_limit = fps_limit
        self.screen_capture.set_fps_limit(fps_limit)
        
        # Initialize game state
        self.is_connected = False
        self.frame_count = 0
        self.last_reset_time = time.time()
        
        # Try to connect to the game
        self.connect()
    
    def connect(self) -> bool:
        """
        Connect to the game by finding its window.
        
        Returns:
            bool: True if connection is successful, False otherwise.
        """
        try:
            self.is_connected = self.screen_capture.find_game_window()
            if self.is_connected:
                print(f"Successfully connected to {self.screen_capture.window_name}")
                print(f"Game resolution: {self.screen_capture.get_resolution()}")
            else:
                print(f"Could not find game window: {self.screen_capture.window_name}")
            return self.is_connected
        except Exception as e:
            print(f"Error connecting to game: {e}")
            self.is_connected = False
            return False
    
    def get_frame(self, resize: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Get the current game frame.
        
        Args:
            resize: Optional (width, height) to resize the captured image.
            
        Returns:
            np.ndarray: The captured frame as a numpy array.
        """
        if not self.is_connected:
            if not self.connect():
                raise RuntimeError("Not connected to the game")
                
        frame = self.screen_capture.capture_screen(resize)
        self.frame_count += 1
        return frame
    
    def take_action(self, action: Any, duration: float = 0.1) -> None:
        """
        Execute an action in the game.
        
        Args:
            action: Action to execute (format depends on your action space).
            duration: How long to hold the action in seconds.
        """
        if not self.is_connected:
            if not self.connect():
                raise RuntimeError("Not connected to the game")
                
        self.input_controller.execute_action(action, duration)
    
    def reset_game(self) -> bool:
        """
        Reset the game to starting state (navigate menus to start a new match).
        
        Returns:
            bool: True if reset was successful, False otherwise.
        """
        try:
            # Press Escape to access menu (if in game)
            self.input_controller.execute_action("PAUSE", 0.1)
            time.sleep(0.5)
            
            # This is a simplified menu navigation and will need to be 
            # customized based on the actual game menu structure
            
            # Example navigation - adjust based on actual menu layouts
            menu_actions = [
                "DOWN",   # Navigate to "Exit to Main Menu"
                "DOWN",
                "ATTACK", # Select "Exit to Main Menu"
                "ATTACK", # Confirm
                "UP",     # Navigate to "Play"
                "ATTACK", # Select "Play"
                "DOWN",   # Navigate to "Arcade" or preferred mode
                "ATTACK", # Select mode
                "ATTACK", # Select character (first character)
                "ATTACK", # Confirm character
            ]
            
            # Delays between menu actions
            delays = [0.3] * len(menu_actions)
            
            # Execute menu navigation
            self.input_controller.navigate_menu(menu_actions, delays)
            
            self.last_reset_time = time.time()
            return True
            
        except Exception as e:
            print(f"Error resetting game: {e}")
            return False
    
    def save_frame(self, frame: np.ndarray, folder: str = "data/recordings") -> str:
        """
        Save a frame to disk for debugging or analysis.
        
        Args:
            frame: The frame to save.
            folder: Folder to save the frame in.
            
        Returns:
            str: Path to the saved frame.
        """
        import cv2
        
        # Create folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{folder}/frame_{timestamp}_{self.frame_count}.png"
        
        # Save the frame
        cv2.imwrite(filename, frame)
        
        return filename
    
    def close(self) -> None:
        """
        Close the game interface and release resources.
        """
        # Release any held keys
        self.input_controller.stop_continuous_action()
        
        # Add any other cleanup here
        print("Game interface closed")


# For testing purposes
if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    
    # Create game interface
    game = GameInterface()
    
    # Test frame capture
    print("Capturing frames...")
    
    frames = []
    try:
        # Capture 5 frames with 0.2s delay
        for i in range(5):
            frame = game.get_frame()
            frames.append(frame)
            
            # Save the frame
            filepath = game.save_frame(frame)
            print(f"Saved frame to {filepath}")
            
            time.sleep(0.2)
            
        # Test actions
        print("Testing actions...")
        game.take_action("RIGHT", 0.5)
        time.sleep(0.2)
        game.take_action("JUMP", 0.2)
        time.sleep(0.2)
        game.take_action(["JUMP", "ATTACK"], 0.3)
        
    finally:
        # Close the interface
        game.close()
    
    # Display captured frames
    fig, axes = plt.subplots(1, len(frames), figsize=(15, 5))
    for i, frame in enumerate(frames):
        if len(frames) > 1:
            ax = axes[i]
        else:
            ax = axes
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Frame {i+1}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()