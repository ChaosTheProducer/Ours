import torch
import torch.nn as nn

class RotationPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=4):
        super(RotationPredictor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(inplace=False),
            nn.Conv3d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=False),
            nn.Conv3d(
                in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=False),
        )
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim=64),
            nn.ReLU(),
            nn.Linear(hidden_dim=64, hidden=128),
            nn.ReLU(),
            nn.Linear(hidden=128, num_classes=4),  
        )

    def forward(self, x):
        conv = self.conv(x)
        fc = torch.flatten(conv, start_dim=1)  # 展平输入
        return self.fc(conv)


