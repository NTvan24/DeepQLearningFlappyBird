# Deep Q-Learning for Flappy Bird

## Overview
This project implements Deep Q-Learning (DQN) to train an AI agent to play Flappy Bird using PyTorch and OpenAI Gymnasium.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install numpy matplotlib pillow gymnasium torch flappy-bird-gymnasium
```

## File Structure
```
DeepQLearning-FlappyBird/
│── model.py         # Defines the neural network for DQN
│── train.py         # Training script for the agent
│── test.py          # Runs a trained model on Flappy Bird
│── utils.py         # Helper functions
│── README.md        # Project documentation
```

## Training the Agent
Run the following command to start training:

```bash
python train.py
```

This will train the AI using Deep Q-Learning and save the model weights.

## Testing the Trained Model
Once training is complete, you can test the model using:

```bash
python test.py
```

## Implementation Details
- Uses **Deep Q-Network (DQN)** with experience replay and target network updates.
- Neural network architecture is defined in `model.py`.
- Uses **flappy-bird-gymnasium** environment for Flappy Bird simulation.

## Results
The model improves over time, avoiding pipes more effectively after sufficient training episodes. Performance can be visualized using matplotlib plots.
- **Maximum score achieved by the AI: 51**

## Future Improvements
- Implement **Double DQN** to reduce overestimation bias.
- Use **Dueling DQN** for better action selection.
- Experiment with **Prioritized Experience Replay**.

## Credits
- Based on the Deep Q-Learning algorithm.
- Uses OpenAI Gymnasium and PyTorch for reinforcement learning.

## License
This project is open-source under the MIT License.

