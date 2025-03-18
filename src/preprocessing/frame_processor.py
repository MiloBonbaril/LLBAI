import numpy as np
import cv2
from collections import deque

class FrameProcessor:
    """
    Processes raw game frames for neural network input.
    
    This class handles the transformation of raw screenshots from Lethal League Blaze
    into a format suitable for neural network processing, including resizing,
    grayscale conversion, normalization, and frame stacking.
    """
    
    def __init__(self, resize_shape=(256, 144), stack_size=4):
        """
        Initialize the frame processor.
        
        Args:
            resize_shape (tuple): Target resolution (height, width) for processed frames
            stack_size (int): Number of consecutive frames to stack
        """
        self.resize_shape = resize_shape
        self.stack_size = stack_size
        self.frame_stack = deque(maxlen=stack_size)
        self._initialize_frame_stack()
    
    def _initialize_frame_stack(self):
        """Initialize the frame stack with blank (zero) frames."""
        # Create blank frames with correct dimensions (height, width)
        # cv2.resize expects (width, height) but numpy arrays are (height, width)
        blank_frame = np.zeros((self.resize_shape[1], self.resize_shape[0]), dtype=np.float32)
        for _ in range(self.stack_size):
            self.frame_stack.append(blank_frame.copy())
    
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame (numpy.ndarray): Raw BGR image from screen capture
            
        Returns:
            numpy.ndarray: Processed frame (grayscale, resized, normalized)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to target shape - cv2.resize expects (width, height)
        resized = cv2.resize(gray, (self.resize_shape[0], self.resize_shape[1]))
        
        # Normalize pixel values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def add_to_stack(self, frame):
        """
        Process a frame and add it to the frame stack.
        
        Args:
            frame (numpy.ndarray): Raw BGR image from screen capture
            
        Returns:
            numpy.ndarray: Updated frame stack as a stacked array
        """
        processed_frame = self.process_frame(frame)
        self.frame_stack.append(processed_frame)
        return self.get_stacked_frames()
    
    def get_stacked_frames(self):
        """
        Get the current frame stack as a stacked array.
        
        Returns:
            numpy.ndarray: Stacked frames with shape (stack_size, height, width)
        """
        return np.array(self.frame_stack)
    
    def reset(self):
        """Reset the frame stack to all zeros."""
        self._initialize_frame_stack()
