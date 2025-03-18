import os
import time
import cv2
import numpy as np
import argparse
from pathlib import Path

# Import game interface components
from game_interface.game_interface import GameInterface


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Lethal League Blaze AI Agent")
    parser.add_argument("--window", type=str, default="Lethal League Blaze",
                        help="Window name of the game")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frame rate limit for screen capture")
    parser.add_argument("--record", action="store_true",
                        help="Record gameplay frames to disk")
    parser.add_argument("--output", type=str, default="data/recordings",
                        help="Output directory for recorded frames")
    parser.add_argument("--resize", type=int, nargs=2, default=None,
                        help="Resize frames to WxH (e.g. --resize 256 224)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Recording duration in seconds")
    return parser.parse_args()


def create_directories(directories):
    """Create directories if they don't exist."""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def test_game_interface(args):
    """Test the game interface components."""
    print(f"Testing game interface with window name: {args.window}")
    
    # Create game interface
    game = GameInterface(window_name=args.window, fps_limit=args.fps)
    
    # Check if connected
    if not game.is_connected:
        print("Failed to connect to the game. Make sure it's running.")
        return
    
    # Record frames if requested
    if args.record:
        create_directories([args.output])
        record_gameplay(game, args)
    else:
        # Just test basic functionality
        test_basic_functionality(game, args)
    
    # Close the interface
    game.close()


def record_gameplay(game, args):
    """Record gameplay frames to disk."""
    print(f"Recording {args.duration} seconds of gameplay to {args.output}...")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < args.duration:
            # Capture frame
            frame = game.get_frame(resize=args.resize)
            
            # Save frame
            filepath = game.save_frame(frame, folder=args.output)
            
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                print(f"Recorded {frame_count} frames ({frame_count/elapsed:.1f} FPS)")
            
    except KeyboardInterrupt:
        print("Recording stopped by user.")
    
    elapsed = time.time() - start_time
    print(f"Recorded {frame_count} frames in {elapsed:.1f} seconds ({frame_count/elapsed:.1f} FPS)")


def test_basic_functionality(game, args):
    """Test basic functionality of the game interface."""
    print("Testing basic functionality...")
    
    # Get current resolution
    resolution = game.screen_capture.get_resolution()
    print(f"Game resolution: {resolution}")
    
    # Capture a test frame
    frame = game.get_frame(resize=args.resize)
    print(f"Captured frame shape: {frame.shape}")
    
    # Test basic actions
    print("Testing basic actions...")
    actions_to_test = [
        "LEFT", "RIGHT", "UP", "DOWN", "JUMP", "ATTACK", 
        ["JUMP", "ATTACK"], ["LEFT", "JUMP"]
    ]
    
    for action in actions_to_test:
        action_name = action if isinstance(action, str) else "+".join(action)
        print(f"Executing action: {action_name}")
        game.take_action(action, duration=0.2)
        time.sleep(0.3)
    
    print("Basic functionality tests complete!")


if __name__ == "__main__":
    args = parse_args()
    # Wait for user to focus on game window
    print("Focus on the game window. Testing will start in 3 seconds...")
    time.sleep(3)
    test_game_interface(args)