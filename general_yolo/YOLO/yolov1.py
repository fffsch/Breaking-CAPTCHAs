import torch
import torch.nn as nn

"""
Architecture config for the YOLOv1 model, specified as a list of tuples.
Tuple: (kernel_size, num_filters, stride, padding)
String "M": MaxPool layer
List: A list of tuples, followed by an integer, represents repeated blocks.
"""
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], # Repeat 4 times
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2], # Repeat 2 times
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    """
    Standard building block for the YOLOv1 architecture.
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class YoloV1(nn.Module):
    """
    The main YOLOv1 model class, built from the architecture config.
    """
    def __init__(self, in_channels=3, S=7, B=2, C=20):
        super(YoloV1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(architecture_config)
        self.fcs = self._create_fcs(S, B, C)

    def forward(self, x):
        x = self.darknet(x)
        # Flatten the output for the fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        # Reshape to S x S x (C + B*5)
        x = x.reshape(-1, self.S, self.S, self.C + self.B * 5)
        return x

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                    )
                ]
                in_channels = x[1]
            elif type(x) == str and x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0] # Tuple
                conv2 = x[1] # Tuple
                num_repeats = x[2] # Integer

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, S, B, C):
        # The 7x7x1024 output from darknet is flattened to 7*7*1024
        # This is then passed through linear layers.
        # The article uses a 4096 intermediate layer.
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)), 
        )