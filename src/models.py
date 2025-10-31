from torch import nn
import torch
import torch.nn.functional as F

class CNN_network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv2d(4, 4, (1, 1), stride=1, padding=(0, 0))

        self.conv1_M = nn.Conv2d(4, 4, (4, 1), stride=1, padding=(0, 0))
        self.conv1_m = nn.Conv2d(4, 4, (3, 1), stride=1, padding=(0, 0))

        self.conv2 = nn.Conv2d(4, 4, (2, 4), stride=2)

        self.fc1 = nn.Linear(24, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)

        x_major = x[:, :, 0:4, :]
        x_minor = x[:, :, 4:8, :]

        x_major = F.relu(self.conv1_M(x_major))
        x_minor = F.relu(self.conv1_m(x_minor))

        x = torch.cat((x_major, x_minor), dim=2)

        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x