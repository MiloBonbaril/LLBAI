import unittest
import torch
import numpy as np
import os
import sys
import coverage
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.cnn_model import LLBAgent

# Create a coverage object
cov = coverage.Coverage(source=['src.models.cnn_model'])

class TestLLBAgent(unittest.TestCase):
    """Test cases for the LLBAgent neural network model."""
    
    @classmethod
    def setUpClass(cls):
        """Start coverage tracking before any tests."""
        cov.start()
    
    @classmethod
    def tearDownClass(cls):
        """Stop coverage tracking after all tests."""
        cov.stop()
        cov.save()
        print("\nCoverage Report:")
        cov.report(show_missing=True)
    
    def setUp(self):
        """Set up test environment before each test."""
        # Define test parameters
        self.stack_size = 4
        self.frame_height = 144
        self.frame_width = 256
        self.num_actions = 10
        self.batch_size = 5
        
        # Create model
        self.input_shape = (self.stack_size, self.frame_height, self.frame_width)
        self.model = LLBAgent(self.input_shape, self.num_actions)
        
        # Create test input (single sample)
        self.test_input_single = torch.rand(self.stack_size, self.frame_height, self.frame_width)
        
        # Create test input (batch)
        self.test_input_batch = torch.rand(self.batch_size, self.stack_size, self.frame_height, self.frame_width)
    
    def test_model_initialization(self):
        """Test if model initializes with correct parameters."""
        self.assertEqual(self.model.input_shape, self.input_shape)
        self.assertEqual(self.model.num_actions, self.num_actions)
        self.assertIsInstance(self.model.conv1, torch.nn.Conv2d)
        self.assertIsInstance(self.model.fc1, torch.nn.Linear)
        self.assertIsInstance(self.model.policy_head, torch.nn.Linear)
        self.assertIsInstance(self.model.value_head, torch.nn.Linear)
    
    def test_forward_single_sample(self):
        """Test forward pass with a single sample."""
        # Set model to eval mode
        self.model.eval()
        
        # Forward pass
        action_logits, state_value = self.model(self.test_input_single)
        
        # Check output shapes
        self.assertEqual(action_logits.shape, (1, self.num_actions))
        self.assertEqual(state_value.shape, (1, 1))
    
    def test_forward_batch(self):
        """Test forward pass with a batch of samples."""
        # Set model to eval mode
        self.model.eval()
        
        # Forward pass
        action_logits, state_value = self.model(self.test_input_batch)
        
        # Check output shapes
        self.assertEqual(action_logits.shape, (self.batch_size, self.num_actions))
        self.assertEqual(state_value.shape, (self.batch_size, 1))
    
    def test_get_action_probs(self):
        """Test get_action_probs method."""
        # Set model to eval mode
        self.model.eval()
        
        # Get action probabilities
        action_probs = self.model.get_action_probs(self.test_input_batch)
        
        # Check output shape
        self.assertEqual(action_probs.shape, (self.batch_size, self.num_actions))
        
        # Check that probabilities sum to 1 for each sample
        sums = action_probs.sum(dim=1)
        for sum_val in sums:
            self.assertAlmostEqual(sum_val.item(), 1.0, places=6)
        
        # Check that all values are between 0 and 1
        self.assertTrue(torch.all(action_probs >= 0).item())
        self.assertTrue(torch.all(action_probs <= 1).item())
    
    def test_get_q_values(self):
        """Test get_q_values method."""
        # Set model to eval mode
        self.model.eval()
        
        # Get Q-values
        q_values = self.model.get_q_values(self.test_input_batch)
        
        # Check output shape
        self.assertEqual(q_values.shape, (self.batch_size, self.num_actions))
    
    def test_save_load(self, tmp_path='./tmp_model.pth'):
        """Test model save and load functionality."""
        import os
        
        # Get original parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Save model
        self.model.save(tmp_path)
        
        # Modify model parameters to ensure load actually changes them
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        # Verify parameters changed
        for name, param in self.model.named_parameters():
            self.assertFalse(torch.allclose(param, original_params[name]))
        
        # Load model
        self.model.load(tmp_path)
        
        # Verify parameters restored to original values
        for name, param in self.model.named_parameters():
            self.assertTrue(torch.allclose(param, original_params[name]))
        
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    def test_conv_output_size(self):
        """Test the _get_conv_output_size method."""
        # Get the output size
        output_size = self.model._get_conv_output_size()
        
        # Check that it's a positive integer
        self.assertIsInstance(output_size, int)
        self.assertGreater(output_size, 0)


if __name__ == '__main__':
    try:
        # Start with coverage
        cov.start()
        unittest.main(exit=False)
    finally:
        # Stop coverage and generate report
        cov.stop()
        cov.save()
        print("\nDetailed Coverage Report:")
        coverage_percentage = cov.report(show_missing=True)
        print(f"Total coverage: {coverage_percentage:.2f}%")
        # Generate HTML report if desired
        try:
            cov.html_report(directory='htmlcov')
            print("HTML coverage report generated in 'htmlcov' directory")
        except Exception as e:
            print(f"Could not generate HTML report: {e}")