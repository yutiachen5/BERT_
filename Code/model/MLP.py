from sklearn.neural_network import MLPRegressor
from torch import nn


class MLPRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self,
                logger,
                x):
        logger.info(self.layers)
        return self.layers(x)
