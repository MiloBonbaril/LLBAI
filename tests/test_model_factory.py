import unittest
import os
import torch
import torch.optim as optim
import shutil
import numpy as np
import sys
import coverage
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.model_factory import ModelFactory
from src.models.cnn_model import LLBAgent

# Create a coverage object
cov = coverage.Coverage(source=['src.models.model_factory'])

class TestModelFactory(unittest.TestCase):
    """Test cases for the ModelFactory class."""
    
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
        self.config = {
            "input_shape": (4, 144, 256),
            "num_actions": 8,
            "use_cuda": False  # Force CPU for testing
        }
        
        # Create model factory
        self.model_factory = ModelFactory(self.config)
        
        # Define test directory for checkpoints
        self.test_checkpoint_dir = './test_checkpoints'
        os.makedirs(self.test_checkpoint_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test directory
        if os.path.exists(self.test_checkpoint_dir):
            shutil.rmtree(self.test_checkpoint_dir)
    
    def test_create_model(self):
        """Test creating a model."""
        # Create model
        model = self.model_factory.create_model()
        
        # Check model type
        self.assertIsInstance(model, LLBAgent)
        
        # Check model parameters
        self.assertEqual(model.input_shape, self.config["input_shape"])
        self.assertEqual(model.num_actions, self.config["num_actions"])
        
        # Check device
        for param in model.parameters():
            self.assertEqual(param.device, torch.device('cpu'))
    
    def test_save_load_checkpoint(self):
        """Test saving and loading a checkpoint."""
        # Create model and optimizer
        model = self.model_factory.create_model()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create test data
        test_episode = 10
        test_rewards = [1.0, 2.0, 3.0]
        
        # Save checkpoint
        checkpoint_path = self.model_factory.save_checkpoint(
            model, optimizer, test_episode, test_rewards, self.test_checkpoint_dir
        )
        
        # Check file exists
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Modify model to ensure loading actually changes parameters
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        # Load checkpoint
        loaded_model, loaded_optimizer, loaded_episode, loaded_rewards, loaded_config = \
            self.model_factory.load_checkpoint(checkpoint_path, model, optimizer)
        
        # Check loaded model parameters match original
        for name, param in loaded_model.named_parameters():
            self.assertTrue(torch.allclose(param, original_params[name]))
        
        # Check loaded episode and rewards
        self.assertEqual(loaded_episode, test_episode)
        self.assertEqual(loaded_rewards, test_rewards)
        
        # Check loaded config
        self.assertEqual(loaded_config, self.config)
    
    def test_load_checkpoint_new_model(self):
        """Test loading a checkpoint into a new model."""
        # Create and save a model
        model = self.model_factory.create_model()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        checkpoint_path = self.model_factory.save_checkpoint(
            model, optimizer, 5, [1.0], self.test_checkpoint_dir
        )
        
        # Get original parameters
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Load checkpoint into a new model (without passing model argument)
        loaded_model, _, _, _, _ = self.model_factory.load_checkpoint(checkpoint_path)
        
        # Check loaded model parameters match original
        for name, param in loaded_model.named_parameters():
            self.assertTrue(torch.allclose(param, original_params[name]))
    
    def test_load_nonexistent_checkpoint(self):
        """Test loading a non-existent checkpoint."""
        # Try to load non-existent checkpoint
        with self.assertRaises(FileNotFoundError):
            self.model_factory.load_checkpoint('nonexistent_checkpoint.pth')


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
            print("HTML coverage report generated in 'htmlcov' directory")
        except Exception as e:
            print(f"Could not generate HTML report: {e}")