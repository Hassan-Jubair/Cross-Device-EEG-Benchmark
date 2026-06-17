# run_final_evaluation.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.autograd import Function
import scipy.stats as stats
import warnings
import copy

warnings.filterwarnings('ignore')

data = np.load('/content/drive/MyDrive/cross_device_eeg_aligned_FINAL.npz')
X = data['X']
y = data['y']
domains = data['domains']

initial_trials = X.shape[0]
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
valid_mask = ~(np.all(X == 0, axis=(1, 2)))
X = X[valid_mask]
y = y[valid_mask]
domains = domains[valid_mask]

class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class SimpleEEGConvNet(nn.Module):
    def __init__(self, channels=20, time_steps=300):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, 16, kernel_size=32, padding=15)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=16, padding=7)
        self.flatten = nn.Flatten()
        
        test_tensor = torch.zeros(1, channels, time_steps)
        out = self.pool(self.relu(self.conv1(test_tensor)))
        out = self.pool(self.relu(self.conv2(out)))
        self.fc1 = nn.Linear(out.view(1, -1).size(1), 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class EEGNetLite(nn.Module):
    def __init__(self, channels=20, time_steps=300):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding='same')
        self.bn1 = nn.BatchNorm2d(8)
        self.depthwise = nn.Conv2d(8, 16, (channels, 1), groups=8)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.separable = nn.Conv2d(16, 16, (1, 16), padding='same', groups=16)
        self.pointwise = nn.Conv2d(16, 16, 1)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.flatten = nn.Flatten()
        
        test_tensor = torch.zeros(1, 1, channels, time_steps)
        out = self.bn1(self.conv1(test_tensor))
        out = self.bn2(self.depthwise(out))
        out = self.pool1(out)
        out = self.pointwise(self.separable(out))
        out = self.pool2(out)
        self.fc = nn.Linear(out.view(1, -1).size(1), 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn1(self.conv1(x))
        x = torch.relu(self.bn2(self.depthwise(x)))
        x = self.pool1(x)
        x = torch.relu(self.pointwise(self.separable(x)))
        x = self.pool2(x)
        return self.fc(self.flatten(x))

class ShallowConvNetLite(nn.Module):
    def __init__(self, channels=20, time_steps=300):
        super().__init__()
        self.conv_time = nn.Conv2d(1, 40, (1, 25))
        self.conv_spat = nn.Conv2d(40, 40, (channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.flatten = nn.Flatten()
        
        test_tensor = torch.zeros(1, 1, channels, time_steps)
        out = self.pool(torch.square(self.bn(self.conv_spat(self.conv_time(test_tensor)))))
        self.fc = nn.Linear(out.view(1, -1).size(1), 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn(self.conv_spat(self.conv_time(x)))
        x = self.pool(torch.square(x))
        x = torch.log(torch.clamp(x, min=1e-6))
        return self.fc(self.flatten(x))

class DANN(nn.Module):
    def __init__(self, channels=20, time_steps=300):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=32, padding=15),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(16, 32, kernel_size=16, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten()
        )
        
        test_tensor = torch.zeros(1, channels, time_steps)
        flat_size = self.feature_extractor(test_tensor).size(1)
        
        self.class_classifier = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x, alpha=0.0):
        features = self.feature_extractor(x)
        class_preds = self.class_classifier(features)
        reverse_features = GRL.apply(features, alpha)
        domain_preds = self.domain_classifier(reverse_features)
        return class_preds, domain_preds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unique_domains = np.unique(domains)

results = {
    "SimpleCNN": [],
    "EEGNet": [],
    "ShallowConvNet": [],
    "DANN": []
}

domain_results = {d: {} for d in unique_domains}

max_epochs = 100
patience = 10

for test_domain in unique_domains:
    
    train_mask = (domains != test_domain)
    test_mask = (domains == test_domain)
    
    X_train_full = X[train_mask]
    y_train_full = y[train_mask]
    
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    X_train_t = torch.tensor(X_train_sub, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train_sub, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val_sub, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_sub, dtype=torch.long).to(device)
    
    X_test_t = torch.tensor(X[test_mask], dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y[test_mask], dtype=torch.long).to(device)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)
    
    class_counts = np.bincount(y_train_sub)
    weights = len(y_train_sub) / (len(class_counts) * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    for model_name, ModelClass in [("SimpleCNN", SimpleEEGConvNet), ("EEGNet", EEGNetLite), ("ShallowConvNet", ShallowConvNetLite)]:
        model = ModelClass().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_loss = float('inf')
        counter = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        for epoch in range(max_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    val_loss += criterion(model(batch_X), batch_y).item()
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break
                    
        model.load_state_dict(best_model_wts)
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                _, preds = torch.max(model(batch_X), 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
        acc = accuracy_score(all_targets, all_preds) * 100
        results[model_name].append(acc)
        domain_results[test_domain][model_name] = acc

    target_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=True, drop_last=True)
    model = DANN().to(device)
    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(max_epochs):
        model.train()
        len_dataloader = min(len(train_loader), len(target_loader))
        source_iter = iter(train_loader)
        target_iter = iter(target_loader)
        
        for i in range(len_dataloader):
            p = float(i + epoch * len_dataloader) / max_epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            source_data, source_label = next(source_iter)
            domain_label_src = torch.zeros(len(source_data), dtype=torch.long).to(device)
            class_pred, domain_pred_src = model(source_data, alpha)
            loss_s_label = class_criterion(class_pred, source_label)
            loss_s_domain = domain_criterion(domain_pred_src, domain_label_src)
            
            target_data, _ = next(target_iter)
            domain_label_tgt = torch.ones(len(target_data), dtype=torch.long).to(device)
            _, domain_pred_tgt = model(target_data, alpha)
            loss_t_domain = domain_criterion(domain_pred_tgt, domain_label_tgt)
            
            loss = loss_s_label + loss_s_domain + loss_t_domain
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                class_preds, _ = model(batch_X, alpha=0.0)
                val_loss += class_criterion(class_preds, batch_y).item()
                
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
                
    model.load_state_dict(best_model_wts)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            class_preds, _ = model(batch_X, alpha=0.0)
            _, preds = torch.max(class_preds, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds) * 100
    results["DANN"].append(acc)
    domain_results[test_domain]["DANN"] = acc

models_list = ["SimpleCNN", "EEGNet", "ShallowConvNet", "DANN"]
print("Per-Domain Results Table")
header = f"{'Held-out Dataset':<20} | " + " | ".join([f"{m:<15}" for m in models_list])
print(header)
for d in unique_domains:
    row = f"Domain {d:<13} | " + " | ".join([f"{domain_results[d][m]:<15.2f}" for m in models_list])
    print(row)

best_baseline_name = None
best_baseline_mean = 0

print("\nFinal Statistical Results")
for name, accs in results.items():
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    ci = stats.t.interval(0.95, df=len(accs)-1, loc=mean_acc, scale=stats.sem(accs))
    print(f"{name}: Mean = {mean_acc:.2f}%, SD = {std_acc:.2f}%, 95% CI = [{ci[0]:.2f}%, {ci[1]:.2f}%]")
    
    if name != "DANN" and mean_acc > best_baseline_mean:
        best_baseline_mean = mean_acc
        best_baseline_name = name

dann_accs = results["DANN"]
base_accs = results[best_baseline_name]

t_stat, p_val = stats.ttest_rel(dann_accs, base_accs)
diffs = np.array(dann_accs) - np.array(base_accs)
cohens_d = np.mean(diffs) / np.std(diffs, ddof=1)

print(f"\nPaired t-test (DANN vs {best_baseline_name}):")
print(f"p-value = {p_val:.4f}")
print(f"Cohen's d = {cohens_d:.2f}")