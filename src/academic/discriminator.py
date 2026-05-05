import torch
from torch import nn


class GRL(torch.autograd.Function):
    """Gradient Reversal Layer with fixed lambda_=0.1."""

    @staticmethod
    def forward(ctx, x, lambda_=0.1):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class DomainDiscriminator(nn.Module):
    """Domain head for synthetic(0) vs real(1) discrimination."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, p4):
        # Domain-adversarial branch from arXiv 2020:
        # "Optimal domain adaptive object detection with self-training and adversarial-based approach".
        # Expects p4 shape: (batch, 256, 40, 40).
        x = self.pool(p4).flatten(1)
        x = GRL.apply(x, 0.1)
        x = self.act(self.bn1(self.fc1(x)))
        x = self.act(self.bn2(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x
