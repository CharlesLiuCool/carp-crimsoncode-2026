import torch.nn as nn


class MergedModel(nn.Module):
    """
    Lightweight 3-feature diabetes risk model.
    Architecture: 3 -> 16 -> 8 -> 1  (~209 parameters)

    Features: Age, BMI, Glucose
    Output:   raw logit (apply sigmoid for probability)

    Must stay in sync with hospital_client/backend/model.py.
    """

    def __init__(self, n_features=3, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)
