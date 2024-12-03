# discriminator.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(100, 128)  # Adjust input dimensions as needed
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)  # Output layer, probability for classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output as probability
        return x
