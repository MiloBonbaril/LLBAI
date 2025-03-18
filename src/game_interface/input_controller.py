import time
from typing import List, Dict, Union, Optional, Tuple
import threading


class InputController:
    """
    Controls game inputs for Lethal League Blaze.
    Uses pydirectinput for more reliable input simulation in games.
    """
    
    def __init__(self):
        """
        Initialize the input controller with default key mappings.
        """
        try:
            import pydirectinput
            self.input = pydirectinput
        except ImportError:
            print("pydirectinput not installed. Falling back to pyautogui.")
            print("Install pydirectinput with: pip install pydirectinput")
            import pyautogui
            self.input = pyautogui
            
        # Configure for game input
        self.input.PAUSE = 0.01  # Short delay between actions
        
        # Default key mappings for Lethal League Blaze
        self.key_map = {
            'LEFT': 'left',
            'RIGHT': 'right',
            'UP': 'up',
            'DOWN': 'down',
            'JUMP': 'space',  # Typically Space for jump
            'ATTACK': 'c',  # Typically C for attack/hit
            'GRAB': 'z',  # Typically Z for GRAB
            'BUNT': 'x',     # Typically X for bunt
            'TAUNT': 'f',    # Typically S for taunt
            'PAUSE': 'escape'  # Escape for pause
        }
        
        # Action cooldowns to prevent input flooding
        self.last_action_time = 0
        self.min_action_interval = 0.05  # 50ms between actions
        
        # For multi-key presses
        self.pressed_keys = set()
        self.action_thread = None
        self.stop_thread = False
    
    def set_key_mapping(self, new_mapping: Dict[str, str]) -> None:
        """
        Update the key mapping.
        
        Args:
            new_mapping: Dictionary mapping action names to keyboard keys.
        """
        self.key_map.update(new_mapping)
    
    def _press_key(self, key: str) -> None:
        """
        Press a key and remember it's pressed.
        
        Args:
            key: The key to press.
        """
        self.input.keyDown(key)
        self.pressed_keys.add(key)
    
    def _release_key(self, key: str) -> None:
        """
        Release a key and remove from pressed keys.
        
        Args:
            key: The key to release.
        """
        self.input.keyUp(key)
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
    
    def _release_all_keys(self) -> None:
        """
        Release all currently pressed keys.
        """
        for key in list(self.pressed_keys):
            self._release_key(key)
    
    def execute_action(self, action: Union[str, List[str]], duration: float = 0.1) -> None:
        """
        Execute a game action or combination of actions.
        
        Args:
            action: String or list of strings representing actions to perform.
            duration: How long to hold the keys down in seconds.
        """
        # Convert single action to list
        if isinstance(action, str):
            action = [action]
            
        # Ensure minimum time between actions
        current_time = time.time()
        if current_time - self.last_action_time < self.min_action_interval:
            pass # pass to test the speed of the test
            # time.sleep(self.min_action_interval - (current_time - self.last_action_time))
        
        # Release any previously pressed keys
        self._release_all_keys()
        
        # Press the new keys
        for act in action:
            if act in self.key_map:
                self._press_key(self.key_map[act])
        
        # Wait for duration
        #time.sleep(duration)
        
        # Release the keys
        self._release_all_keys()
        
        self.last_action_time = time.time()
    
    def execute_sequence(self, actions: List[Tuple[Union[str, List[str]], float]]) -> None:
        """
        Execute a sequence of actions with specified durations.
        
        Args:
            actions: List of (action, duration) tuples.
        """
        for action, duration in actions:
            self.execute_action(action, duration)
    
    def start_continuous_action(self, action: Union[str, List[str]]) -> None:
        """
        Start a continuous action that runs in the background until stopped.
        
        Args:
            action: String or list of strings representing actions to perform continuously.
        """
        # Stop any existing continuous action
        self.stop_continuous_action()
        
        # Convert single action to list
        if isinstance(action, str):
            action = [action]
        
        # Press the new keys
        for act in action:
            if act in self.key_map:
                self._press_key(self.key_map[act])
    
    def stop_continuous_action(self) -> None:
        """
        Stop any continuous action that is currently running.
        """
        self._release_all_keys()
    
    def navigate_menu(self, menu_actions: List[str], delays: List[float]) -> None:
        """
        Execute a series of menu navigation actions with delays.
        
        Args:
            menu_actions: List of menu navigation actions.
            delays: List of delays between actions.
        """
        for action, delay in zip(menu_actions, delays):
            self.execute_action(action, 0.1)
            time.sleep(delay)


# For testing purposes
if __name__ == "__main__":
    # Create input controller
    controller = InputController()
    
    # Wait for user to focus on game window
    print("Focus on the game window. Testing will start in 3 seconds...")
    time.sleep(3)
    
    # Test basic actions
    print("Testing basic movements...")
    
    # Move right
    print("Moving right...")
    controller.execute_action("RIGHT", 0.5)
    time.sleep(0.5)
    
    # Move left
    print("Moving left...")
    controller.execute_action("LEFT", 0.5)
    time.sleep(0.5)
    
    # Jump
    print("Jumping...")
    controller.execute_action("JUMP", 0.2)
    time.sleep(0.5)
    
    # Attack
    print("Attacking...")
    controller.execute_action("ATTACK", 0.2)
    time.sleep(0.5)
    
    # Combo: Jump attack
    print("Jump attacking...")
    controller.execute_action(["JUMP", "ATTACK"], 0.3)
    time.sleep(0.5)
    
    print("Input testing complete!")