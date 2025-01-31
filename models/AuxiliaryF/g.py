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
        
        # 修复全连接层的定义
        self.fc = nn.Sequential(
            nn.Linear(16384, hidden_dim),  # 输入特征 -> 隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),       # 隐藏层 -> 更深的隐藏层
            nn.ReLU(),
            nn.Linear(128, num_classes),      # 最终分类
        )

    def forward(self, x):
        #print("x:",x.shape)
        conv = self.conv(x)  # 卷积操作
        #print("conv:",conv.shape)
        fc_input = torch.flatten(conv, start_dim=1)  # 展平输入
        #print("fc_in",fc_input.shape)
        return self.fc(fc_input)



