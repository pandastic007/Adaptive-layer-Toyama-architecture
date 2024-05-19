import torch
import torch.nn as nn

class PianoClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PianoClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.mean(dim=2)  # Average over time dimension
        return self.fc(x)
