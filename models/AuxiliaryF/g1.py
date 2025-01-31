import torch
import torch.nn as nn

# 分类旋转角度
class RotationPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=4):
        super(RotationPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)  # 输出单个值，表示旋转角度
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # 展平输入
        return self.fc(x)
