import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
from moabb import datasets
from moabb.paradigms import MotorImagery
from collections import Counter
import numpy as np
from EEGNets.EEGNet_residual import ResEEG

dataset = datasets.BNCI2014_001()
paradigm = MotorImagery()
subjects = dataset.subject_list
print(f"Subjects: {subjects}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_results = []
all_loss_histories = []
all_val_acc_histories = []

for test_subj in subjects:
    train_subs = [s for s in subjects if s != test_subj]
    X_train, y_train, _ = paradigm.get_data(dataset=dataset, subjects=train_subs)
    X_val, y_val, _ = paradigm.get_data(dataset=dataset, subjects=[test_subj])
    
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)

    class_counts = Counter(y_train)
    total_samples = len(y_train)
    weights = torch.tensor([1.0 / class_counts[i] for i in range(len(class_counts))]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    # normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
    
    C, T = X_train.shape[1], X_train.shape[2]
    n_classes = len(np.unique(y_train))
    X_train = X_train[:, np.newaxis, :, :]
    X_val = X_val[:, np.newaxis, :, :]
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128, shuffle=False)
    model = ResEEG(n_chan=C, n_cls=n_classes, F=8, T=T).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    best_val_acc = 0
    
    loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        loss_history.append(avg_train_loss)
        
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        val_acc_history.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

    all_results.append(best_val_acc)
    all_loss_histories.append(loss_history)
    all_val_acc_histories.append(val_acc_history)
    print(f"Best Val Acc (Subject {test_subj}): {best_val_acc:.4f}")

mean_acc = np.mean(all_results)
std_acc = np.std(all_results)
for i, acc in enumerate(all_results, start=1):
    print(f"Subject {i}: {acc:.4f}")
print(f"Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}")

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

colors = plt.cm.Set3(np.linspace(0, 1, len(subjects)))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
epochs_range = range(1, num_epochs + 1)

for i, (loss_h, val_h) in enumerate(zip(all_loss_histories, all_val_acc_histories)):
    ax1.plot(epochs_range, loss_h, color=colors[i], label=f'Subject {subjects[i]}')
    ax2.plot(epochs_range, val_h, color=colors[i], label=f'Subject {subjects[i]}')

import matplotlib.ticker as ticker
ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax1.set_title('Training Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True)

ax2.set_title('Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True)

plt.tight_layout()
subfloder = Path("result_imgs")
subfloder.mkdir(parents=True, exist_ok=True)
plt.savefig(subfloder / "EEG_residual_Competition_IV.png")

plt.show()
