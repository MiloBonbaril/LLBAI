import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_processed_frame(original_frame, processed_frame, fig_size=(10, 5)):
    """
    Visualize original and processed frames side by side.
    
    Args:
        original_frame: Raw BGR image
        processed_frame: Processed grayscale image (normalized 0-1)
        fig_size: Figure size as (width, height)
    """
    # Convert to display format
    original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    processed_display = (processed_frame * 255).astype(np.uint8)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    
    # Plot original image
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Frame')
    axes[0].axis('off')
    
    # Plot processed image
    axes[1].imshow(processed_display, cmap='gray')
    axes[1].set_title('Processed Frame')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_frame_stack(frame_stack, fig_size=(10, 3)):
    """
    Visualize a stack of frames in a grid.
    
    Args:
        frame_stack: Numpy array of stacked frames (stack_size, height, width)
        fig_size: Figure size as (width, height)
    """
    stack_size = frame_stack.shape[0]
    
    # Create plot
    fig, axes = plt.subplots(1, stack_size, figsize=fig_size)
    
    # Plot each frame in the stack
    for i in range(stack_size):
        frame = (frame_stack[i] * 255).astype(np.uint8)
        axes[i].imshow(frame, cmap='gray')
        axes[i].set_title(f'Frame {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def test_frame_processor_interactive(frame_processor, screen_capture, num_frames=20, delay=0.1):
    """
    Interactive test of frame processor with live game capture.
    
    Args:
        frame_processor: Initialized FrameProcessor instance
        screen_capture: Screen capture object that returns game frames
        num_frames: Number of frames to process
        delay: Delay between frames in seconds
    """
    import time
    
    frame_processor.reset()
    plt.figure(figsize=(12, 8))
    
    for i in range(num_frames):
        # Capture frame
        frame = screen_capture.capture_screen()
        
        # Process and add to stack
        frame_processor.add_to_stack(frame)
        stack = frame_processor.get_stacked_frames()
        
        # Display
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Original Game Frame')
        plt.axis('off')
        
        # Display each frame in the stack
        for j in range(stack.shape[0]):
            plt.subplot(2, 2, j+1)
            plt.imshow(stack[j], cmap='gray')
            plt.title(f'Stacked Frame {j}')
            plt.axis('off')
        
        plt.suptitle(f'Frame Processing Test {i+1}/{num_frames}')
        plt.tight_layout()
        plt.draw()
        plt.pause(delay)
        
        time.sleep(delay)
