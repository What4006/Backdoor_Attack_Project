import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

class LiraGenerator(nn.Module):
    def __init__(self, input_channels=3, epsilon=8/255.0):
        super(LiraGenerator, self).__init__()
        self.epsilon = epsilon
        
        self.down1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # 1. 编码
        d1 = self.down1(x)
        d2 = self.down2(d1)
        
        # 2. 瓶颈处理
        r = self.res_blocks(d2)
        
        # 3. 解码
        u1 = self.up1(r)
        noise = self.up2(u1)
        
        # 4. 缩放噪声 (LIRA 的核心)
        # 将噪声严格限制在 [-epsilon, +epsilon] 之间
        noise = noise * self.epsilon
        
        # 5. 叠加噪声
        # 最终图像 = 原图 + 噪声 (并截断到 0-1 之间)
        poisoned_x = torch.clamp(x + noise, 0, 1)
        
        return poisoned_x, noise