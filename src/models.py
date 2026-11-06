import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class SimpleCNN(nn.Module):
    def __init__(self, channels=20, time_steps=300):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=25, padding=12, bias=False),
            nn.BatchNorm1d(32), nn.ELU(), nn.AvgPool1d(4), nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.ELU(), nn.AvgPool1d(4), nn.Dropout(0.3),
            nn.Flatten()
        )
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, channels, time_steps)).shape[1]
        self.classifier = nn.Sequential(nn.Linear(flat_dim, 64), nn.ELU(), nn.Dropout(0.3), nn.Linear(64, 2))
    def forward_features(self, x): return self.features(x)
    def forward(self, x): return self.classifier(self.features(x))

class EEGNet82(nn.Module):
    def __init__(self, channels=20, time_steps=300, F1=8, D=2, F2=16, dropout=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 51), padding=(0, 25), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 15), padding=(0, 7), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False), nn.BatchNorm2d(F2),
            nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout), nn.Flatten()
        )
        with torch.no_grad():
            flat_dim = self.block2(self.block1(torch.zeros(1, 1, channels, time_steps))).shape[1]
        self.classifier = nn.Linear(flat_dim, 2)
    def forward_features(self, x): return self.block2(self.block1(x.unsqueeze(1)))
    def forward(self, x): return self.classifier(self.forward_features(x))

class ShallowConvNet(nn.Module):
    def __init__(self, channels=20, time_steps=300, n_filters=40, dropout=0.5):
        super().__init__()
        self.conv_time = nn.Conv2d(1, n_filters, (1, 25), bias=False)
        self.conv_spat = nn.Conv2d(n_filters, n_filters, (channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.drop = nn.Dropout(dropout)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, time_steps)
            flat_dim = self.pool(torch.square(self.bn(self.conv_spat(self.conv_time(dummy))))).view(1, -1).shape[1]
        self.classifier = nn.Linear(flat_dim, 2)
    def forward_features(self, x):
        x = self.bn(self.conv_spat(self.conv_time(x.unsqueeze(1))))
        return self.drop(torch.log(torch.clamp(self.pool(torch.square(x)), min=1e-6)).flatten(1))
    def forward(self, x): return self.classifier(self.forward_features(x))

class Deep4Net(nn.Module):
    def __init__(self, channels=20, time_steps=300, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), bias=False), nn.Conv2d(25, 25, (channels, 1), bias=False),
            nn.BatchNorm2d(25), nn.ELU(), nn.MaxPool2d((1, 3), stride=(1, 3)), nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(nn.Conv2d(25, 50, (1, 10), bias=False), nn.BatchNorm2d(50), nn.ELU(), nn.MaxPool2d((1, 3), stride=(1, 3)), nn.Dropout(dropout))
        self.block3 = nn.Sequential(nn.Conv2d(50, 100, (1, 10), bias=False), nn.BatchNorm2d(100), nn.ELU(), nn.MaxPool2d((1, 3), stride=(1, 3)), nn.Dropout(dropout))
        self.block4 = nn.Sequential(nn.Conv2d(100, 200, (1, 4), bias=False), nn.BatchNorm2d(200), nn.ELU(), nn.MaxPool2d((1, 2), stride=(1, 2)), nn.Dropout(dropout), nn.Flatten())
        with torch.no_grad():
            flat_dim = self.block4(self.block3(self.block2(self.block1(torch.zeros(1, 1, channels, time_steps))))).shape[1]
        self.classifier = nn.Linear(flat_dim, 2)
    def forward_features(self, x): return self.block4(self.block3(self.block2(self.block1(x.unsqueeze(1)))))
    def forward(self, x): return self.classifier(self.forward_features(x))

class EEGConformer(nn.Module):
    def __init__(self, channels=20, time_steps=300, emb_size=40, depth=2, n_heads=4, dropout=0.3):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, emb_size, (1, 25), bias=False), nn.Conv2d(emb_size, emb_size, (channels, 1), bias=False),
            nn.BatchNorm2d(emb_size), nn.ELU(), nn.AvgPool2d((1, 45), stride=(1, 10)), nn.Dropout(dropout)
        )
        enc = nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_heads, dim_feedforward=emb_size*2, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=depth)
        with torch.no_grad():
            dummy = self.patch_embed(torch.zeros(1, 1, channels, time_steps)).squeeze(2).transpose(1, 2)
            flat_dim = dummy.shape[1] * dummy.shape[2]
        self.classifier = nn.Sequential(nn.Linear(flat_dim, 64), nn.ELU(), nn.Dropout(dropout), nn.Linear(64, 2))
    def forward_features(self, x): return self.transformer(self.patch_embed(x.unsqueeze(1)).squeeze(2).transpose(1, 2)).flatten(1)
    def forward(self, x): return self.classifier(self.forward_features(x))

class EEGNet_DANN(nn.Module):
    def __init__(self, channels=20, time_steps=300, n_domains=6):
        super().__init__()
        self.backbone = EEGNet82(channels=channels, time_steps=time_steps)
        in_dim = self.backbone.classifier.in_features
        self.class_head = nn.Linear(in_dim, 2)
        self.domain_head = nn.Sequential(nn.Linear(in_dim, 64), nn.ELU(), nn.Dropout(0.25), nn.Linear(64, n_domains))
    def forward_features(self, x): return self.backbone.forward_features(x)
    def forward(self, x, alpha=0.0):
        feats = self.backbone.forward_features(x)
        return self.class_head(feats), self.domain_head(GradReverse.apply(feats, alpha)), feats
