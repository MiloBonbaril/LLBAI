# Lethal League Blaze AI Agent

This project implements a deep learning AI agent that learns to play Lethal League Blaze using vision-based reinforcement learning. The agent sees the game screen and learns to take actions to hit the ball and score points.

## Project Structure

```
lethal-league-agent/
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── data/                    # Data storage
│   ├── recordings/          # Game screen recordings
│   ├── models/              # Saved model checkpoints
│   └── logs/                # Training logs
├── src/                     # Source code
│   ├── game_interface/      # Game interaction modules
│   ├── preprocessing/       # Image preprocessing modules
│   ├── models/              # Neural network architectures
│   ├── learning/            # Reinforcement learning algorithms
│   ├── utils/               # Utility functions
│   └── main.py              # Main training script
└── tests/                   # Unit tests
```

## Installation

1. Clone this repository:
```bash
git clone #https://github.com/yourusername/lethal-league-agent.git
cd lethal-league-agent
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Testing the Game Interface

To test the basic game interface functionality:

```bash
python src/main.py --window "LLBlaze"
```

### Recording Gameplay Frames

To record gameplay frames for analysis:

```bash
python src/main.py --window "LLBlaze" --record --duration 60 --output data/recordings
```

### Resizing Frames

To capture frames at a lower resolution:

```bash
python src/main.py --window "LLBlaze" --record --resize 160 120
```

## Development Roadmap

- [x] Phase 1: Game Interface
  - [x] Screen Capture
  - [x] Input Controller
  - [x] Basic Integration

- [ ] Phase 2: Image Preprocessing
  - [ ] Frame Processing
  - [ ] Frame Stacking

- [ ] Phase 3: Neural Network Architecture
  - [ ] CNN Model Implementation

- [ ] Phase 4: Reinforcement Learning System
  - [ ] Replay Memory
  - [ ] DQN Implementation
  - [ ] Reward System

- [ ] Phase 5: Training Pipeline
  - [ ] Main Training Loop
  - [ ] Logging and Visualization

## Technical Details

### Game Interface

The game interface module provides:
- Screen capture at configurable frame rates
- Keyboard/controller input simulation
- Game state management (reset, menu navigation)

### Image Preprocessing (Upcoming)

Planned preprocessing features:
- Frame downscaling
- Grayscale conversion
- Frame stacking
- Normalization

### Neural Network (Upcoming)

Planned network architecture:
- CNN backbone for visual feature extraction
- Policy/value heads for reinforcement learning
- Support for DQN and policy gradient methods

### Reinforcement Learning (Upcoming)

Planned RL components:
- Experience replay memory
- DQN implementation with target networks
- Reward function based on survival time and game events

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Team Reptile for creating Lethal League Blaze
- DeepMind for pioneering vision-based reinforcement learning for games