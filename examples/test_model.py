import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.cnn_model import LLBAgent
from src.preprocessing.frame_processor import FrameProcessor
from src.game_interface.screen_capture import ScreenCapture


def visualize_model_activations(model, input_tensor):
    """
    Visualize activations of each convolutional layer in the model.
    
    Args:
        model (LLBAgent): The model to visualize
        input_tensor (torch.Tensor): Input to the model
    """
    # Set model to eval mode
    model.eval()
    
    # Get activations
    activations = {}
    
    # Define hooks to capture activations
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    hook1 = model.conv1.register_forward_hook(get_activation('conv1'))
    hook2 = model.conv2.register_forward_hook(get_activation('conv2'))
    hook3 = model.conv3.register_forward_hook(get_activation('conv3'))
    
    # Forward pass
    with torch.no_grad():
        model(input_tensor)
    
    # Remove hooks
    hook1.remove()
    hook2.remove()
    hook3.remove()
    
    # Create a figure with 2x2 grid for overall layout
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Model Activations Visualization", fontsize=16)
    
    # Plot input
    ax_input = axs[0, 0]
    ax_input.set_title("Input (first frame)")
    # Add a check for the tensor shape and reshape if needed
    input_image = input_tensor[0, 0].cpu().numpy()
    print(f"Input image shape: {input_image.shape}")
    
    # If the input is 1D, reshape it to 2D using the known dimensions
    if len(input_image.shape) == 1:
        input_image = input_image.reshape(256, 144)
    
    ax_input.imshow(input_image, cmap='gray')
    ax_input.axis('off')
    
    # Function to plot feature maps
    def plot_feature_maps(activation, ax, title, max_maps=16):
        ax.set_title(title)
        ax.axis('off')  # Turn off the outer subplot axes
        
        # Get the first 16 feature maps (or less if fewer available)
        num_maps = min(max_maps, activation.shape[1])
        
        # Create a grid for feature maps
        grid_size = int(np.ceil(np.sqrt(num_maps)))
        
        feature_maps = activation[0].cpu().numpy()
        
        # Create a grid of subplots within the given axis
        inner_grid = fig.add_gridspec(grid_size, grid_size, wspace=0.1, hspace=0.1, 
                                      left=ax.get_position().x0, right=ax.get_position().x1,
                                      bottom=ax.get_position().y0, top=ax.get_position().y1)
        
        for i in range(num_maps):
            # Create a proper subplot in the inner grid
            inner_ax = fig.add_subplot(inner_grid[i // grid_size, i % grid_size])
            
            # Ensure the feature map is 2D
            feat_map = feature_maps[i]
            if len(feat_map.shape) == 1:
                # If 1D, try to determine the correct dimensions
                feat_size = int(np.sqrt(feat_map.size))
                if feat_size * feat_size == feat_map.size:
                    feat_map = feat_map.reshape(feat_size, feat_size)
                else:
                    # Skip this feature map if we can't reshape it
                    inner_ax.axis('off')
                    continue
            
            inner_ax.imshow(feat_map, cmap='viridis')
            inner_ax.axis('off')
    
    # Plot feature maps for each layer
    plot_feature_maps(activations['conv1'], axs[0, 1], "First Convolutional Layer")
    plot_feature_maps(activations['conv2'], axs[1, 0], "Second Convolutional Layer")
    plot_feature_maps(activations['conv3'], axs[1, 1], "Third Convolutional Layer")
    
    plt.subplots_adjust(top=0.9)
    plt.show()


def visualize_action_distribution(model, input_tensor):
    """
    Visualize the action probability distribution.
    
    Args:
        model (LLBAgent): The model to visualize
        input_tensor (torch.Tensor): Input to the model
    """
    # Set model to eval mode
    model.eval()
    
    # Get action probabilities
    with torch.no_grad():
        action_probs = model.get_action_probs(input_tensor)
    
    # Convert to numpy
    action_probs = action_probs.cpu().numpy()[0]
    
    # Action names (customize based on your action space)
    action_names = [
        'No Action',
        'Move Left',
        'Move Right',
        'Jump',
        'Attack',
        'Jump+Attack',
        'Jump+Left',
        'Jump+Right',
        'Grab',
        'Bunt'
    ][:len(action_probs)]
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(action_names, action_probs)
    
    # Add values above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.title('Action Probability Distribution')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def main():
    """Test model with game capture."""
    # Parameters
    input_shape = (4, 256, 144)
    num_actions = 10
    
    # Create model
    model = LLBAgent(input_shape, num_actions)
    print(f"Created model: {model.__class__.__name__}")
    print(f"Input shape: {input_shape}")
    print(f"Number of actions: {num_actions}")
    
    # Initialize preprocessor and screen capture
    processor = FrameProcessor(resize_shape=(input_shape[1], input_shape[2]), stack_size=input_shape[0])
    screen_capture = ScreenCapture(window_name="LLBlaze")
    
    print("Capturing frames for testing model...")
    
    # Capture and process frames
    for _ in range(input_shape[0]):
        frame = screen_capture.capture_screen()
        processor.add_to_stack(frame)
    
    # Get frame stack as tensor
    frame_stack = processor.get_stacked_frames()
    
    # Print shape to debug
    print(f"Frame stack shape: {frame_stack.shape}")
    
    # Ensure proper shape before converting to tensor
    input_tensor = torch.tensor(frame_stack, dtype=torch.float32)
    
    # Check if we need to reshape the input tensor for the CNN model
    if len(input_tensor.shape) == 2:
        # If we got a 2D tensor, reshape to [batch, channels, height, width]
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif len(input_tensor.shape) == 3:
        # If we got a 3D tensor, add batch dimension
        input_tensor = input_tensor.unsqueeze(0)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Visualize model activations
    print("Visualizing model activations...")
    visualize_model_activations(model, input_tensor)
    
    # Visualize action distribution
    print("Visualizing action probability distribution...")
    visualize_action_distribution(model, input_tensor)
    
    print("Model test complete.")


if __name__ == "__main__":
    main()