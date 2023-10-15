import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, input_dim, projection_dims):
        super(Projector, self).__init__()
        layers = []
        if input_dim == 1024:
            layers.append(nn.Conv2d(1024, 2048, kernel_size=2, stride=2))
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            layers.append(nn.Flatten(1))
        for i in range(len(projection_dims) - 2):
            layers.append(nn.Linear(projection_dims[i], projection_dims[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(projection_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(projection_dims[-2], projection_dims[-1], bias=False))
        self.projector = nn.Sequential(*layers)
    
    def forward(self, x):
        embedding = self.projector(x)
        return embedding