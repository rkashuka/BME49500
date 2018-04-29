import torch.nn as nn

kernel_size = 3


class ColorNet:
    # Define model
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2)
        )
