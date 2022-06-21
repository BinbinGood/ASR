from torch import nn

__all__ = ['ConvStack']


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, input_dim):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride)
        self.act = nn.GELU()
        # self.dropout = nn.Dropout()
        self.output_dim = (input_dim - self.kernel_size) // self.stride + 1

    def forward(self, x, x_len):
        x = self.conv(x)
        x = self.act(x)
        # x = self.dropout(x)
        x_len = ((x_len - self.kernel_size) / self.stride + 1).int()
        return x, x_len


class ConvStack(nn.Module):
    """具有堆叠卷积层的卷积组

    :param feat_size: 输入音频的特征大小
    :type feat_size: int
    :param conv_out_channels: 卷积层输出大小
    :type conv_out_channels: int
    """

    def __init__(self, feat_size, conv_out_channels):
        super().__init__()
        self.conv1 = ConvBn(in_channels=1,
                            out_channels=conv_out_channels,
                            kernel_size=3,
                            stride=2,
                            input_dim=feat_size)
        self.conv2 = ConvBn(in_channels=conv_out_channels,
                            out_channels=conv_out_channels,
                            kernel_size=3,
                            stride=2,
                            input_dim=self.conv1.output_dim)
        self.output_dim = self.conv2.output_dim * conv_out_channels

    def forward(self, x, x_len):
        """
        x: shape [B, D, T]
        x_len : shape [B]
        """
        # [B, D, T] -> [B, T, D]
        x = x.permute(0, 2, 1)
        # [B, T, D] -> [B, C=1, T, D]
        x = x.unsqueeze(1)
        x, x_len = self.conv1(x, x_len)
        x, x_len = self.conv2(x, x_len)
        # 将数据从卷积特征映射转换为向量序列
        x = x.permute(0, 2, 1, 3)  # [B, T, C, D]
        x = x.reshape([x.shape[0], x.shape[1], -1])  # [B, T, C*D]
        return x, x_len
