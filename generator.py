# generator.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        # Define layers of the generator model
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)  # Specify output_dim accordingly
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Add activation function if needed
        return x
