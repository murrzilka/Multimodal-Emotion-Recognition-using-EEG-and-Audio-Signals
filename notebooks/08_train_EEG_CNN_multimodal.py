import torch.nn as nn
import torch.nn.functional as F
from save_metrics import save_metrics_to_json
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

class EEGNet(nn.Module):
    def __init__(self, num_classes=2):
        super(EEGNet, self).__init__()
        # Input: (batch, 1, 32, 512)
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32))
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, (32, 1))
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 8))
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(16, 32, (1, 16), padding=(0, 8))
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.25)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = F.elu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioCNN, self).__init__()
        # Input: (batch, 1, 128, time)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MultimodalModel(nn.Module):
    def __init__(self, eeg_model, audio_model, num_classes=2, fusion_size=64):
        super(MultimodalModel, self).__init__()
        self.eeg_model = eeg_model
        self.audio_model = audio_model
        self.eeg_model.fc = nn.Identity()
        self.audio_model.fc = nn.Identity()
        # merging and adding a common classifier.
        self.fusion = nn.Sequential(
            nn.Linear(32 + 128, fusion_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_size, num_classes)
        )

    def forward(self, eeg, audio):
        eeg_feat = self.eeg_model(eeg)
        audio_feat = self.audio_model(audio)
        combined = torch.cat([eeg_feat, audio_feat], dim=1)
        out = self.fusion(combined)
        return out

#
# Tensors loading
#
X_eeg = np.load("../data/processed/eeg_tensors.npy")
y_eeg = np.load("../data/processed/eeg_labels_val.npy")

X_audio = np.load("../data/processed/audio_spectrograms.npy")
y_audio = np.load("../data/processed/audio_labels_val.npy")

# Creating artificial (fixed) pairs
def create_pairs(X1, y1, X2, y2, n_pairs_per_class=600):
    np.random.seed(42)
    classes = np.unique(y1)
    X1_pairs, X2_pairs, y_pairs = [], [], []
    for cls in classes:
        idx1 = np.where(y1 == cls)[0]
        idx2 = np.where(y2 == cls)[0]
        if len(idx1) < n_pairs_per_class:
            idx1 = np.random.choice(idx1, n_pairs_per_class, replace=True)
        else:
            idx1 = np.random.choice(idx1, n_pairs_per_class, replace=False)
        if len(idx2) < n_pairs_per_class:
            idx2 = np.random.choice(idx2, n_pairs_per_class, replace=True)
        else:
            idx2 = np.random.choice(idx2, n_pairs_per_class, replace=False)
        for i in range(n_pairs_per_class):
            X1_pairs.append(X1[idx1[i]])
            X2_pairs.append(X2[idx2[i]])
            y_pairs.append(cls)
    X1_pairs = np.array(X1_pairs)
    X2_pairs = np.array(X2_pairs)
    y_pairs = np.array(y_pairs)
    perm = np.random.permutation(len(y_pairs))
    return X1_pairs[perm], X2_pairs[perm], y_pairs[perm]


X_eeg_pairs, X_audio_pairs, y_pairs = create_pairs(X_eeg, y_eeg, X_audio, y_audio, n_pairs_per_class=600)
print("Pairs created:", len(y_pairs))

# Convert to tensors
X_eeg_t = torch.tensor(X_eeg_pairs, dtype=torch.float32)
X_audio_t = torch.tensor(X_audio_pairs, dtype=torch.float32)
y_t = torch.tensor(y_pairs, dtype=torch.long)


#
# Training and evaluation functions
#
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for eeg, audio, labels in dataloader:
        eeg, audio, labels = eeg.to(device), audio.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(eeg, audio) if isinstance(model, MultimodalModel) else model(eeg)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc

def evaluate_with_probs(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for eeg, audio, labels in dataloader:
            eeg, audio, labels = eeg.to(device), audio.to(device), labels.to(device)
            outputs = model(eeg, audio) if isinstance(model, MultimodalModel) else model(eeg)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    return total_loss / len(dataloader), acc, f1, np.array(all_probs), np.array(all_preds), np.array(all_labels)

#
# Cross-validation function
#
def run_cross_validation(X_eeg, X_audio, y, model_class, model_params, n_folds=5, batch_size=32, epochs=30, lr=0.001):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    fold_accs, fold_f1s = [], []
    all_y_true, all_y_pred, all_y_prob = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_eeg, y)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")
        train_dataset = TensorDataset(X_eeg[train_idx], X_audio[train_idx], y[train_idx])
        val_dataset = TensorDataset(X_eeg[val_idx], X_audio[val_idx], y[val_idx])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        # Creating models
        if model_class == 'eeg':
            model = EEGNet(num_classes=2).to(device)
        elif model_class == 'audio':
            model = AudioCNN(num_classes=2).to(device)
        elif model_class == 'multimodal':
            eeg_model = EEGNet(num_classes=2).to(device)
            audio_model = AudioCNN(num_classes=2).to(device)
            model = MultimodalModel(eeg_model, audio_model, num_classes=2).to(device)
        else:
            raise ValueError("Unknown model class")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_probs, best_preds, best_labels = None, None, None

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1, val_probs, val_preds, val_labels = evaluate_with_probs(
                model, val_loader, criterion, device
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_probs = val_probs
                best_preds = val_preds
                best_labels = val_labels

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        fold_accs.append(best_val_acc)
        fold_f1s.append(best_val_f1)
        all_y_true.extend(best_labels)
        all_y_pred.extend(best_preds)
        all_y_prob.extend(best_probs)

        print(f"Fold {fold + 1} best val Acc: {best_val_acc:.4f}, F1: {best_val_f1:.4f}")

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)

    return mean_acc, std_acc, mean_f1, std_f1, np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob)

# --------------------------------------------
# Cross-validation for three models
# --------------------------------------------
models_to_run = {
    'EEG-only': 'eeg',
    'Audio-only': 'audio',
    'Multimodal': 'multimodal'
}

results = {}
for name, model_type in models_to_run.items():
    print(f"\n\n===== Model evaluation: {name} =====")
    mean_acc, std_acc, mean_f1, std_f1, y_true, y_pred, y_prob = run_cross_validation(
        X_eeg_t, X_audio_t, y_t,
        model_class=model_type,
        model_params={},
        n_folds=5,
        batch_size=32,
        epochs=50,
        lr=0.001
    )
    results[name] = {
        'acc': mean_acc,
        'std_acc': std_acc,
        'f1': mean_f1,
        'std_f1': std_f1,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

#
# Results
#
print("\n===== Final results =====")
for name, metrics in results.items():
    print(f"{name:15} | Acc: {metrics['acc']:.4f} ± {metrics['std_acc']:.4f} | F1: {metrics['f1']:.4f} ± {metrics['std_f1']:.4f}")

save_metrics_to_json(results, "../results/metrics.json")