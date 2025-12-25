import torch
import torch.nn as nn

# 下采样块
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0, max_pooling=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2) if max_pooling else None
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.dropout:
            x = self.dropout(x)
        skip = x
        if self.maxpool:
            x = self.maxpool(x)
        return x, skip

# 上采样块
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, dropout_prob=0.15):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.att_dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        if self.att_dropout:
            out = self.att_dropout(out)
        return out

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, dropout_prob=0.15):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.att_dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        out = self.sigmoid(out)
        if self.att_dropout:
            out = self.att_dropout(out)
        return out

# 双分支注意力模块
class DualBranchAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 通道注意力分支
        self.channel_attention = ChannelAttention(in_channels, reduction)
        # 空间注意力分支
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # 应用通道注意力
        channel_out = self.channel_attention(x)
        # 应用空间注意力
        spatial_out = self.spatial_attention(x)
        # 融合两种注意力
        out = x * channel_out * spatial_out
        return out

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, n_filters=32):
        """
        :param n_channels: 输入图像通道数，默认为3(RGB)
        :param n_classes: 输出类别数，默认为1(二分类分割)
        :param n_filters: 初始滤波器数量，默认为32
        """
        super().__init__()

        # 编码器路径 3→32→64→128→256→512
        self.down1 = DownBlock(n_channels, n_filters)
        self.down2 = DownBlock(n_filters, n_filters * 2)
        self.down3 = DownBlock(n_filters * 2, n_filters * 4)
        self.down4 = DownBlock(n_filters * 4, n_filters * 8)
        self.down5 = DownBlock(n_filters * 8, n_filters * 16)

        # 瓶颈层 - 移除最后的最大池化 512→1024
        self.bottleneck = DownBlock(n_filters * 16, n_filters * 32, dropout_prob=0.4, max_pooling=False)

        # 解码器路径 1024→512→256→128→64→32
        self.up1 = UpBlock(n_filters * 32, n_filters * 16)
        self.up2 = UpBlock(n_filters * 16, n_filters * 8)
        self.up3 = UpBlock(n_filters * 8, n_filters * 4)
        self.up4 = UpBlock(n_filters * 4, n_filters * 2)
        self.up5 = UpBlock(n_filters * 2, n_filters)

        # 输出层
        self.outc = nn.Conv2d(n_filters, n_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码器路径
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x4, skip4 = self.down4(x3)
        x5, skip5 = self.down5(x4)
        # 瓶颈层
        x6, skip6 = self.bottleneck(x5)
        # 解码器路径
        x = self.up1(x6, skip5)
        x = self.up2(x, skip4)
        x = self.up3(x, skip3)
        x = self.up4(x, skip2)
        x = self.up5(x, skip1)

        x = self.outc(x)
        x = self.sigmoid(x)
        return x  

class DBA_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, n_filters=32):
        """
        改进版UNet:在bottleneck和up1层使用注意力      
        :param n_channels: 输入图像通道数，默认为3(RGB)
        :param n_classes: 输出类别数，默认为1(二分类分割)
        :param n_filters: 初始滤波器数量，默认为32
        """
        super().__init__()

        # 编码器路径 3→32→64→128→256→512
        self.down1 = DownBlock(n_channels, n_filters)
        self.down2 = DownBlock(n_filters, n_filters * 2)
        self.down3 = DownBlock(n_filters * 2, n_filters * 4)
        self.down4 = DownBlock(n_filters * 4, n_filters * 8)
        self.down5 = DownBlock(n_filters * 8, n_filters * 16)

        # 瓶颈层 - 移除最后的最大池化 512→1024
        self.bottleneck = DownBlock(n_filters * 16, n_filters * 32, dropout_prob=0.4, max_pooling=False)
        # 仅在瓶颈层使用注意力
        self.bottleneck_attention = DualBranchAttention(n_filters * 32)

        # 解码器路径 1024→512→256→128→64→32
        self.up1 = UpBlock(n_filters * 32, n_filters * 16)
        # 仅在up1层使用注意力
        self.up1_attention = DualBranchAttention(n_filters * 16)
        
        self.up2 = UpBlock(n_filters * 16, n_filters * 8)
        self.up3 = UpBlock(n_filters * 8, n_filters * 4)
        self.up4 = UpBlock(n_filters * 4, n_filters * 2)
        self.up5 = UpBlock(n_filters * 2, n_filters)

        # 输出层
        self.outc = nn.Conv2d(n_filters, n_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码器路径
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x4, skip4 = self.down4(x3)
        x5, skip5 = self.down5(x4)

        # 瓶颈层
        x6, skip6 = self.bottleneck(x5)  
        x6 = self.bottleneck_attention(x6)  # 仅在瓶颈层应用注意力

        # 解码器路径
        x = self.up1(x6, skip5)
        x = self.up1_attention(x)  # 仅在up1层应用注意力

        x = self.up2(x, skip4)
        x = self.up3(x, skip3)
        x = self.up4(x, skip2)
        x = self.up5(x, skip1)

        x = self.outc(x)
        x = self.sigmoid(x)
        return x
