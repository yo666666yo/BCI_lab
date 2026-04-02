"""Control-mode training pipeline for MultiDecoder4Control.

Trains a shared encoder with separate intensity (force/speed) and direction
decoders for BCI robotic control. The control signal is computed as:
    control_vector = intensity_magnitude * direction_probability

Uses MOABB motor imagery datasets with simulated control labels derived from
the original class labels.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EEGNets'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'))

from MultiDecoder_imply import MultiDecoder4Control
from multidecoder_config import get_control_config, ExperimentConfig


def generate_control_labels(y, n_original_classes, n_intensity=3, n_direction=5):
    """Generate simulated intensity and direction labels from original class labels.

    For motor imagery:
    - Direction: derived from the original class (each MI class = a direction)
    - Intensity: simulated as a function of class + random variation
      (in practice, this would come from EMG or force sensors)
    """
    direction = y.copy()
    if n_original_classes < n_direction:
        direction = direction % n_direction

    # Simulate intensity: base intensity per class + noise
    np.random.seed(42)
    base_intensity = {i: i % n_intensity for i in range(n_original_classes)}
    intensity = np.array([base_intensity[int(c)] for c in y], dtype=np.int64)
    # Add some randomness (10% chance of adjacent intensity level)
    noise_mask = np.random.random(len(intensity)) < 0.1
    noise_shift = np.random.choice([-1, 1], size=len(intensity))
    intensity[noise_mask] = np.clip(intensity[noise_mask] + noise_shift[noise_mask],
                                     0, n_intensity - 1)

    return intensity, direction


def train_control_model(config=None):
    """Train MultiDecoder4Control with LOSO cross-validation."""
    if config is None:
        config = get_control_config()

    print("=" * 70)
    print("  Control-Mode Multi-Decoder Training")
    print(f"  Dataset: {config.data.dataset_name}")
    intensity_cfg = config.tasks[0]
    direction_cfg = config.tasks[1]
    print(f"  Intensity: {intensity_cfg.n_classes}-class | "
          f"Direction: {direction_cfg.n_classes}-class")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from moabb import datasets
    from moabb.paradigms import MotorImagery

    dataset_map = {
        'BNCI2014_001': datasets.BNCI2014_001,
        'BNCI2015_004': datasets.BNCI2015_004,
    }
    dataset = dataset_map[config.data.dataset_name]()
    paradigm = MotorImagery()
    subjects = dataset.subject_list

    all_results = {}

    for test_subj in subjects:
        print(f"\n{'─' * 60}")
        print(f"  LOSO: Test subject = {test_subj}")
        print(f"{'─' * 60}")

        train_subs = [s for s in subjects if s != test_subj]
        X_train_raw, y_train_raw, _ = paradigm.get_data(dataset=dataset, subjects=train_subs)
        X_val_raw, y_val_raw, _ = paradigm.get_data(dataset=dataset, subjects=[test_subj])

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train_raw)
        y_val_enc = le.transform(y_val_raw)
        n_classes = len(le.classes_)

        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw.reshape(len(X_train_raw), -1)
                                        ).reshape(X_train_raw.shape)
        X_val = scaler.transform(X_val_raw.reshape(len(X_val_raw), -1)
                                  ).reshape(X_val_raw.shape)

        C, T = X_train.shape[1], X_train.shape[2]

        # Generate control labels
        intensity_train, direction_train = generate_control_labels(
            y_train_enc, n_classes,
            n_intensity=intensity_cfg.n_classes,
            n_direction=direction_cfg.n_classes)
        intensity_val, direction_val = generate_control_labels(
            y_val_enc, n_classes,
            n_intensity=intensity_cfg.n_classes,
            n_direction=direction_cfg.n_classes)

        # To tensors
        X_train_t = torch.FloatTensor(X_train[:, np.newaxis, :, :])
        X_val_t = torch.FloatTensor(X_val[:, np.newaxis, :, :])
        int_train_t = torch.LongTensor(intensity_train)
        dir_train_t = torch.LongTensor(direction_train)
        int_val_t = torch.LongTensor(intensity_val)
        dir_val_t = torch.LongTensor(direction_val)

        train_loader = DataLoader(
            TensorDataset(X_train_t, int_train_t, dir_train_t),
            batch_size=config.training.batch_size, shuffle=True)
        val_loader = DataLoader(
            TensorDataset(X_val_t, int_val_t, dir_val_t),
            batch_size=config.training.batch_size, shuffle=False)

        # Build model
        model = MultiDecoder4Control(
            n_chan=C,
            n_intensity=intensity_cfg.n_classes,
            n_direction=direction_cfg.n_classes,
            F_T=config.encoder.F_T,
            K_T=config.encoder.K_T,
            L=config.encoder.L,
            hidden_dim=config.decoder.hidden_dim,
            dropout=config.decoder.dropout,
        ).to(device)

        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Loss with class weights
        int_counts = Counter(intensity_train)
        dir_counts = Counter(direction_train)
        int_weights = torch.tensor([len(intensity_train) / (len(int_counts) * int_counts[i])
                                     for i in range(intensity_cfg.n_classes)]).float().to(device)
        dir_weights = torch.tensor([len(direction_train) / (len(dir_counts) * dir_counts[i])
                                     for i in range(direction_cfg.n_classes)]).float().to(device)

        criterion_int = nn.CrossEntropyLoss(weight=int_weights)
        criterion_dir = nn.CrossEntropyLoss(weight=dir_weights)

        # Learnable loss balancing
        log_var_int = nn.Parameter(torch.zeros(1, device=device))
        log_var_dir = nn.Parameter(torch.zeros(1, device=device))

        optimizer = torch.optim.Adam(
            list(model.parameters()) + [log_var_int, log_var_dir],
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.training.lr_step_size,
            gamma=config.training.lr_gamma)

        best_combined_acc = 0
        best_state = None
        patience = 0

        for epoch in range(config.training.num_epochs):
            # Train
            model.train()
            total_loss = 0
            n_batch = 0
            for bx, b_int, b_dir in train_loader:
                bx, b_int, b_dir = bx.to(device), b_int.to(device), b_dir.to(device)
                optimizer.zero_grad()
                outputs = model(bx)

                loss_int = criterion_int(outputs['intensity'], b_int)
                loss_dir = criterion_dir(outputs['direction'], b_dir)

                # Uncertainty-weighted loss
                prec_int = torch.exp(-log_var_int)
                prec_dir = torch.exp(-log_var_dir)
                loss = prec_int * loss_int + log_var_int + prec_dir * loss_dir + log_var_dir

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batch += 1

            scheduler.step()

            # Validate
            model.eval()
            int_preds, int_true = [], []
            dir_preds, dir_true = [], []
            with torch.no_grad():
                for bx, b_int, b_dir in val_loader:
                    bx = bx.to(device)
                    outputs = model(bx)
                    int_preds.extend(outputs['intensity'].argmax(1).cpu().numpy())
                    dir_preds.extend(outputs['direction'].argmax(1).cpu().numpy())
                    int_true.extend(b_int.numpy())
                    dir_true.extend(b_dir.numpy())

            int_acc = accuracy_score(int_true, int_preds)
            dir_acc = accuracy_score(dir_true, dir_preds)
            combined_acc = 0.4 * int_acc + 0.6 * dir_acc

            if combined_acc > best_combined_acc:
                best_combined_acc = combined_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/n_batch:.4f} | "
                      f"Int: {int_acc:.3f} | Dir: {dir_acc:.3f} | "
                      f"w_int: {torch.exp(-log_var_int).item():.2f} | "
                      f"w_dir: {torch.exp(-log_var_dir).item():.2f}")

            if patience >= config.training.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Final eval
        if best_state:
            model.load_state_dict(best_state)
            model = model.to(device)

        model.eval()
        int_preds, int_true = [], []
        dir_preds, dir_true = [], []
        with torch.no_grad():
            for bx, b_int, b_dir in val_loader:
                bx = bx.to(device)
                outputs = model(bx)
                int_preds.extend(outputs['intensity'].argmax(1).cpu().numpy())
                dir_preds.extend(outputs['direction'].argmax(1).cpu().numpy())
                int_true.extend(b_int.numpy())
                dir_true.extend(b_dir.numpy())

        int_acc = accuracy_score(int_true, int_preds)
        dir_acc = accuracy_score(dir_true, dir_preds)

        print(f"\n  Subject {test_subj}: Intensity={int_acc:.4f}, Direction={dir_acc:.4f}")

        all_results[test_subj] = {
            'intensity_accuracy': float(int_acc),
            'direction_accuracy': float(dir_acc),
            'combined_accuracy': float(0.4 * int_acc + 0.6 * dir_acc),
        }

    # Aggregate
    print("\n" + "=" * 70)
    print("  CONTROL-MODE AGGREGATE RESULTS")
    print("=" * 70)

    int_accs = [r['intensity_accuracy'] for r in all_results.values()]
    dir_accs = [r['direction_accuracy'] for r in all_results.values()]
    combined = [r['combined_accuracy'] for r in all_results.values()]

    print(f"\n  Intensity:  {np.mean(int_accs):.4f} +/- {np.std(int_accs):.4f}")
    print(f"  Direction:  {np.mean(dir_accs):.4f} +/- {np.std(dir_accs):.4f}")
    print(f"  Combined:   {np.mean(combined):.4f} +/- {np.std(combined):.4f}")

    # Save
    os.makedirs('checkpoints', exist_ok=True)
    with open('checkpoints/results_control.json', 'w') as f:
        json.dump({
            'per_subject': {str(k): v for k, v in all_results.items()},
            'intensity_mean': float(np.mean(int_accs)),
            'direction_mean': float(np.mean(dir_accs)),
            'combined_mean': float(np.mean(combined)),
        }, f, indent=2)

    return all_results


if __name__ == '__main__':
    train_control_model()
