from torch import cat, randn, nn as nn
from torchvision.transforms import CenterCrop


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                 activation_fun=nn.ReLU):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            activation_fun(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            activation_fun(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(
            self,
            initial_in_channels=1,
            initial_out_channels=1,
            features=(64, 128, 256, 512),
            conv_kernel_size=3,
            conv_stride=3,
            up_conv_kernel_size=2,
            up_conv_stride=2,
            max_pool_kernel_size=2,
            max_pool_stride=2,
            end_conv_kernel_size=1,
            end_conv_stride=1
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        features = list(features)

        # Down part of UNet
        # It consists on Conv and Activation Function
        in_channels = initial_in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.pool = nn.MaxPool2d(max_pool_kernel_size, max_pool_stride)

        # Up part of UNet
        # It consists of Conv and Activation Function
        for feature in reversed(features):
            self.up_convs.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=up_conv_kernel_size,
                    stride=up_conv_stride
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Middle part of UNet
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # End part of UNet
        self.final_conv = nn.Conv2d(features[0], initial_out_channels, end_conv_kernel_size)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up_conv, up in zip(self.up_convs, self.ups):
            x = up_conv(x)
            skip_connection = skip_connections.pop(-1)

            crop = CenterCrop(skip_connection.shape[-2::])
            x = crop(x)

            assert skip_connection.shape == x.shape, str(skip_connection.shape) + ":" + str(x.shape)

            concat_skip = cat((skip_connection, x), dim=1)

            x = up(concat_skip)

        return self.final_conv(x)


def test():
    x = randn((3, 1, 162, 162))

    model = UNet(initial_in_channels=1, initial_out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)


if __name__ == "__main__":
    test()
