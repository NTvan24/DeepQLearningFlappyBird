import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import flappy_bird_gymnasium
import gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import randint
import time
import os
import model
import pygame
pygame.quit()  # Đóng tất cả cửa sổ đang mở
pygame.init()  # Khởi tạo lại Pygame

#Environment
env = gymnasium.make("FlappyBird-v0", render_mode="human")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

agent = model.Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=0)

saved_model="model_checkpoint_best_16.21.pth" # Load your best model here
agent.qnetwork_target.load_state_dict(torch.load(saved_model))
# Load Model
agent.qnetwork_local.load_state_dict(torch.load(saved_model))
print("✅ Done!")

num_time_steps=5 #Play 5 times
for i in range(num_time_steps):
    state = env.reset()[0]  # Lấy state ban đầu
    state = np.array(state, dtype=np.float32).reshape(1, -1)
    state = np.array(state, dtype=np.float32)  # Đảm bảo state là np.ndarray
    done = False
    total_reward = 0

    while not done:
        env.render()  # Lấy hình ảnh từ môi trường
        

        # Chọn hành động theo chính sách của agent
        action = agent.act(state, eps=0)  # eps=0 để chọn hành động tối ưu
        
        # Thực hiện hành động
        next_state, reward, done, _,_ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32).reshape(1, -1)
        next_state = np.array(next_state, dtype=np.float32)  # Đảm bảo next_state là np.ndarray
    
        state = next_state
        total_reward += reward

    print(f"Episode {i+1}: Total Reward = {total_reward}")

env.close()  # Đóng môi trường sau khi chơi xong
