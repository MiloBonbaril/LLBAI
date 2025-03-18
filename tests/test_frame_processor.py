import unittest
import numpy as np
import cv2
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.frame_processor import FrameProcessor

class TestFrameProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.processor = FrameProcessor(resize_shape=(84, 84), stack_size=4)
        
        # Create a sample frame (200x150 with random colors)
        self.test_frame = np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8)
        
    def test_process_frame(self):
        """Test if process_frame correctly transforms the input frame."""
        processed = self.processor.process_frame(self.test_frame)
        
        # Check output shape
        self.assertEqual(processed.shape, (84, 84))
        
        # Check normalization (all values should be between 0 and 1)
        self.assertTrue(np.all(processed >= 0))
        self.assertTrue(np.all(processed <= 1))
        
        # Check data type
        self.assertEqual(processed.dtype, np.float32)
    
    def test_add_to_stack(self):
        """Test if add_to_stack correctly updates the frame stack."""
        # Initial stack should be all zeros
        initial_stack = self.processor.get_stacked_frames()
        self.assertEqual(initial_stack.shape, (4, 84, 84))
        self.assertTrue(np.all(initial_stack == 0))
        
        # Add a frame to the stack
        updated_stack = self.processor.add_to_stack(self.test_frame)
        
        # Check that the stack has been updated correctly
        self.assertEqual(updated_stack.shape, (4, 84, 84))
        
        # The first 3 frames should still be zeros
        self.assertTrue(np.all(updated_stack[0:3] == 0))
        
        # The last frame should now contain data (not all zeros)
        self.assertFalse(np.all(updated_stack[3] == 0))
    
    def test_multiple_stack_additions(self):
        """Test adding multiple frames to the stack."""
        # Add 5 frames (with stack size 4 should roll over)
        frames = []
        for i in range(5):
            # Create frames with different colors
            frame = np.ones((150, 200, 3), dtype=np.uint8) * (i * 50)
            frames.append(frame)
            self.processor.add_to_stack(frame)
        
        final_stack = self.processor.get_stacked_frames()
        
        # Stack should still be size 4
        self.assertEqual(final_stack.shape, (4, 84, 84))
        
        # First frame should have been pushed out
        processed_last_frame = self.processor.process_frame(frames[-1])
        self.assertTrue(np.array_equal(final_stack[3], processed_last_frame))
    
    def test_reset(self):
        """Test if reset correctly reinitializes the frame stack."""
        # Add some frames
        for _ in range(3):
            self.processor.add_to_stack(self.test_frame)
        
        # Reset the stack
        self.processor.reset()
        
        # Check if stack is all zeros again
        reset_stack = self.processor.get_stacked_frames()
        self.assertTrue(np.all(reset_stack == 0))

if __name__ == '__main__':
    try:
        import coverage
        cov = coverage.Coverage()
        cov.start()
        unittest.main(exit=False)
        cov.stop()
        cov.save()
        
        print("\nCoverage Report:")
        cov.report()
        
        # Optional: Generate HTML report
        # cov.html_report(directory='coverage_html_report')
        # print("HTML report generated in coverage_html_report directory")
    except ImportError:
        print("Coverage.py not installed. Run 'pip install coverage' to enable coverage reporting.")
        unittest.main()
