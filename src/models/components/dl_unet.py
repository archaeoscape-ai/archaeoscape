from tokenize import Double

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP


class DLUNet(nn.Module):
    def __init__(
        self,
        img_size,
        num_channels,
        num_classes,
        hidden_width=64,
        depth=4,
        pooling=2,
        kernel_size=3,
        bilinear=False,
        pooling_depth=None,
        dilation=[1, 1],
        use_ASPP=True,
        chkpt_path=None,
    ):
        """UNet model for image segmentation, implementing some part of Deep Lab v3.

        Args:
            img_size (int): Input image size.
            num_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            hidden_width (int, optional): Number of filters in the first layer. Defaults to 64.
            depth (int, optional): Depth of the U-Net. Defaults to 4.
            pooling (int, optional): Size of the max pooling kernel. Defaults to 2.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
            bilinear (bool, optional): Whether to use bilinear interpolation in the upsampling layers. Defaults to False.
            pooling_depth (int, optional): Depth of the max pooling layers, afterward dilated convolution are used. If None, no dilated convolution are used Defaults to None.
            dilation (list, optional): Dilation factor for the 2 dilated convolution in each block. Defaults to [1, 1].
        """
        super().__init__()
        self.n_channels = num_channels
        self.n_classes = num_classes
        self.bilinear = bilinear
        if pooling_depth is None:
            pooling_depth = depth
        assert pooling_depth <= depth
        self.pooling_depth = pooling_depth
        self.use_ASPP = use_ASPP

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        # Initial convolution
        self.inc = DoubleConv(num_channels, hidden_width, kernel_size)
        # Down sampling
        for i in range(pooling_depth):
            self.downs.append(
                Down(
                    hidden_width * (2**i),
                    hidden_width * (2 ** (i + 1)),
                    pooling,
                    kernel_size,
                )
            )

        # finish the downsampling with dilated convolutions
        for d in range(pooling_depth, depth):
            i = d - pooling_depth + 1
            self.downs.append(
                DoubleConv(
                    hidden_width * (2**pooling_depth),
                    hidden_width * (2**pooling_depth),
                    kernel_size,
                    dilation=(2**i) * dilation,
                )
            )

        if self.use_ASPP:
            self.ASPP_module = ASPP(
                hidden_width * (2**pooling_depth),
                [12, 24, 36],
                out_channels=256,
            )
        elif isinstance(self.use_ASPP, list):
            self.ASPP_module = ASPP(
                hidden_width * (2**pooling_depth),
                self.use_ASPP,
                out_channels=hidden_width * (2**pooling_depth),
            )
            self.use_ASPP = True

        # Start "upsampling" with dilated convolutions
        for d in range(depth, pooling_depth, -1):
            i = d - pooling_depth + 1
            self.ups.append(
                DoubleConv(
                    hidden_width * (2**pooling_depth),
                    hidden_width * (2**pooling_depth),
                    kernel_size,
                    dilation=(2**i) * dilation,
                )
            )
        # Reduce the number of channels before bilinear upsampling
        if bilinear:
            self.ups.append(
                nn.Conv2d(
                    hidden_width * (2**pooling_depth),
                    hidden_width * (2**pooling_depth) // 2,
                    kernel_size=1,
                    padding="same",
                )
            )
            self.ups.append(nn.ReLU(inplace=True))
        factor = 2 if bilinear else 1
        for i in range(pooling_depth, 0, -1):
            if i == 1:
                # there is no bilinear upsampling afterwards
                factor = 1
            self.ups.append(
                Up(
                    hidden_width * (2 ** (i)),
                    hidden_width * (2 ** (i - 1)) // factor,
                    pooling,
                    kernel_size,
                    bilinear,
                )
            )
        self.outc = OutConv(hidden_width, num_classes)

        if chkpt_path:
            raise NotImplementedError("Checkpointing not implemented yet")

    def forward(self, x):
        """Forward pass of the U-Net architecture.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, height, width).

        Returns:
            dict: A dictionary containing the output logits.
        """
        x_inter = []
        x_inter.append(self.inc(x))
        for down in self.downs:
            x_inter.append(down(x_inter[-1]))
        x_out = x_inter[-1]
        i = 0
        if self.use_ASPP:
            x_out = self.ASPP_module(x_out)

        for up in self.ups:
            if isinstance(up, nn.Conv2d) or isinstance(up, nn.ReLU):
                x_out = up(x_out)
            else:
                x_out = up(x_out, x_inter[-2 - i])
                i += 1
        logits = self.outc(x_out)
        return {"out": logits}

    def use_checkpointing(self):
        """Use checkpointing to reduce memory usage."""
        self.inc = torch.utils.checkpoint(self.inc)
        for i in range(len(self.downs)):
            self.downs[i] = torch.utils.checkpoint(self.downs[i])
        for i in range(len(self.ups)):
            self.ups[i] = torch.utils.checkpoint(self.ups[i])
        self.outc = torch.utils.checkpoint(self.outc)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mid_channels=None,
        dilation=[1, 1],
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                dilation=dilation[0],
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation[1],
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the double convolution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels, out_channels, pooling, kernel_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pooling),
            DoubleConv(in_channels, out_channels, kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Down module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after maxpooling and double convolution.
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(
        self,
        in_channels,
        out_channels,
        pooling,
        kernel_size,
        bilinear=True,
        dilation=None,
    ):
        super().__init__()

        if dilation is None:
            # if bilinear, use the normal convolutions to reduce the number of channels
            if bilinear:
                self.up = nn.Upsample(
                    scale_factor=pooling, mode="bilinear", align_corners=True
                )
                self.conv = DoubleConv(
                    in_channels, out_channels, kernel_size, in_channels // 2
                )
            else:
                self.up = nn.ConvTranspose2d(
                    in_channels,
                    in_channels // 2,
                    kernel_size=pooling,
                    stride=pooling,
                )
                self.conv = DoubleConv(in_channels, out_channels, kernel_size)
        else:
            # no upsampling
            self.up = nn.Identity()
            self.conv = DoubleConv(
                in_channels, out_channels, kernel_size, dilation=dilation
            )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UNet model.

        Args:
            x1 (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            x2 (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Last Conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """Performs a forward pass through the UNet model.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            torch.Tensor: The output tensor from the model.
        """
        return self.conv(x)
