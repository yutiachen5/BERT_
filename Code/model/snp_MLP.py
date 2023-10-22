
from torch import nn


class SNPEmbedding(nn.Module):
    def __init__(self,
                 logger,
                 pretrained_mdl_dir,
                 smiles_dir,
                 downstream_data_dir):
        super().__init__()
        self.logger = logger
        self.pretrained_mdl_dir = pretrained_mdl_dir
        self.smiles_dir = smiles_dir
        self.downstream_data_dir = downstream_data_dir

        self.MLP_predict = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 1)
        )

    def forward(self, x):
        output = self.MLP_predict(x)  # [24, 768] --> [1,768], emb of a cel line
        return output

