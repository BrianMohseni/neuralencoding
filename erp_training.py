from eeg_model import EEGNet, EEGNetConfig

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

data = pd.read_csv("eeg_recording.csv")
X = data[["channel_1", "channel_2", "channel_3", "channel_4"]].values
y = data["space_pressed"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

window_size = 64
stride = 16

X_windows, y_windows = [], []

for start in range(0, len(X) - window_size, stride):
    end = start + window_size
    window = X[start:end]
    label = 1 if y[start:end].max() == 1 else 0
    X_windows.append(window)
    y_windows.append(label)

X_windows = np.array(X_windows)
y_windows = np.array(y_windows)

X_train, X_temp, y_train, y_temp = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42, stratify=y_windows)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class_counts = np.bincount(y_train)
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
sample_weights = weights[y_train_tensor.long()]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, sampler=sampler)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)

config = EEGNetConfig()
model = EEGNet(config)

pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5)

best_val_loss = float('inf')
patience, trigger_times = 25, 0

for epoch in range(50):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    train_loss /= len(train_loader)

    scheduler.step(val_loss)

    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "model.pt")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            break

model.load_state_dict(torch.load("model.pt"))
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = (model(X_batch).squeeze() > 0.5).int()
        all_preds.extend(preds.numpy())
        all_labels.extend(y_batch.numpy())

acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Test Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
