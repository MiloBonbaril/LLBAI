import numpy as np
import cv2
import time
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.game_interface.screen_capture import ScreenCapture
from src.preprocessing.frame_processor import FrameProcessor
from src.utils.visualization import visualize_processed_frame, visualize_frame_stack, test_frame_processor_interactive

def main():
    # Initialize the screen capture
    screen_capture = ScreenCapture(window_name="LLBlaze")
    
    # Initialize the frame processor
    frame_processor = FrameProcessor(resize_shape=(256, 144), stack_size=4)
    
    print("Starting frame processing test...")
    print("Press Ctrl+C to exit.")
    
    try:
        # Basic demonstration
        # Capture a single frame
        frame = screen_capture.capture_screen()
        processed = frame_processor.process_frame(frame)
        
        # Visualize the original and processed frames
        visualize_processed_frame(frame, processed)
        
        # Add multiple frames to the stack
        print("Capturing frames for stack...")
        for _ in range(4):
            frame = screen_capture.capture_screen()
            frame_processor.add_to_stack(frame)
            time.sleep(0.1)  # Small delay between captures
        
        # Get and visualize the frame stack
        stack = frame_processor.get_stacked_frames()
        visualize_frame_stack(stack)
        
        # Run interactive test
        print("Running interactive test...")
        test_frame_processor_interactive(frame_processor, screen_capture, num_frames=30, delay=0.1)
        
    except KeyboardInterrupt:
        print("Test stopped by user.")
    
    print("Frame processing test complete.")

if __name__ == "__main__":
    main()
