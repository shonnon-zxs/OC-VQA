import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch
import torch.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(inplace=False),
            # F.ReLU(inplace=False),
            nn.Dropout(dropout, inplace=False),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
    


if __name__ == '__main__':
    model = SimpleClassifier(10, 20, 2, 0.2)

    print(model)
    x = torch.randn(2, 10)
    logits = model(x)
    print(logits)
