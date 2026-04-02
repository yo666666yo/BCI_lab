"""Evaluation and comparison utilities for multi-decoder EEG models.

Provides:
- Single-decoder baseline training for comparison
- Multi-decoder vs single-decoder accuracy comparison
- Visualization: bar charts, training curves, confusion matrices
- Statistical significance testing (paired t-test)
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EEGNets'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'))


# ─── Single-Decoder Baselines ────────────────────────────────────────────────

def train_single_decoder_baseline(model_cls, model_kwargs, dataset_name='BNCI2014_001',
                                   num_epochs=50, batch_size=64, lr=0.001):
    """Train a single-decoder model as baseline using LOSO-CV.

    Args:
        model_cls: Model class (e.g., ResEEG, EEG_TCNet)
        model_kwargs: Dict of kwargs for model constructor (excluding n_chan/n_cls)
        dataset_name: MOABB dataset name
        num_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate

    Returns:
        Dict with per-subject accuracies
    """
    from moabb import datasets
    from moabb.paradigms import MotorImagery

    dataset_map = {
        'BNCI2014_001': datasets.BNCI2014_001,
        'BNCI2015_004': datasets.BNCI2015_004,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = dataset_map[dataset_name]()
    paradigm = MotorImagery()
    subjects = dataset.subject_list

    results = {}

    for test_subj in subjects:
        train_subs = [s for s in subjects if s != test_subj]
        X_train_raw, y_train_raw, _ = paradigm.get_data(dataset=dataset, subjects=train_subs)
        X_val_raw, y_val_raw, _ = paradigm.get_data(dataset=dataset, subjects=[test_subj])

        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        y_val = le.transform(y_val_raw)
        n_classes = len(le.classes_)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw.reshape(len(X_train_raw), -1)).reshape(X_train_raw.shape)
        X_val = scaler.transform(X_val_raw.reshape(len(X_val_raw), -1)).reshape(X_val_raw.shape)

        C, T = X_train.shape[1], X_train.shape[2]

        X_train_t = torch.FloatTensor(X_train[:, np.newaxis, :, :])
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val[:, np.newaxis, :, :])
        y_val_t = torch.LongTensor(y_val)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

        # Build model
        kwargs = {**model_kwargs, 'n_chan': C}
        if 'n_cls' in model_cls.__init__.__code__.co_varnames:
            kwargs['n_cls'] = n_classes
        elif 'n_class' in model_cls.__init__.__code__.co_varnames:
            kwargs['n_class'] = n_classes
        elif 'N' in model_cls.__init__.__code__.co_varnames:
            kwargs['N'] = n_classes

        model = model_cls(**kwargs).to(device)

        # Class weights
        counts = Counter(y_train)
        weights = torch.tensor([len(y_train) / (n_classes * counts[i])
                                 for i in range(n_classes)]).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

        best_acc = 0
        best_state = None

        for epoch in range(num_epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            preds, true = [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(device)
                    out = model(bx)
                    preds.extend(out.argmax(1).cpu().numpy())
                    true.extend(by.numpy())

            acc = accuracy_score(true, preds)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        results[test_subj] = float(best_acc)
        print(f"  Subject {test_subj}: {best_acc:.4f}")

    return results


# ─── Comparison & Visualization ───────────────────────────────────────────────

def compare_models(results_dict, save_path='result_imgs/comparison.png'):
    """Generate comparison bar chart for multiple models.

    Args:
        results_dict: Dict[model_name, Dict[subject, accuracy]]
        save_path: Path to save the figure
    """
    model_names = list(results_dict.keys())
    subjects = sorted(list(results_dict[model_names[0]].keys()))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Per-subject comparison
    ax1 = axes[0]
    x = np.arange(len(subjects))
    width = 0.8 / len(model_names)

    for i, name in enumerate(model_names):
        accs = [results_dict[name].get(s, 0) for s in subjects]
        ax1.bar(x + i * width, accs, width, label=name, alpha=0.8)

    ax1.set_xlabel('Subject')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Per-Subject Accuracy Comparison')
    ax1.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax1.set_xticklabels([str(s) for s in subjects], rotation=45)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)

    # Aggregate comparison
    ax2 = axes[1]
    means = []
    stds = []
    for name in model_names:
        accs = list(results_dict[name].values())
        means.append(np.mean(accs))
        stds.append(np.std(accs))

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    bars = ax2.bar(model_names, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax2.set_ylabel('Mean Accuracy')
    ax2.set_title('Aggregate Accuracy (Mean +/- Std)')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, m, s in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
                 f'{m:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison chart saved to: {save_path}")


def plot_confusion_matrices(results_json_path, save_path='result_imgs/confusion_matrices.png'):
    """Plot confusion matrices from saved results."""
    with open(results_json_path) as f:
        data = json.load(f)

    subjects = list(data['per_subject'].keys())
    n_subj = len(subjects)

    if n_subj == 0:
        print("No results to plot.")
        return

    # Only plot first 9 subjects max
    n_plot = min(n_subj, 9)
    ncols = min(3, n_plot)
    nrows = (n_plot + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(n_plot):
        subj = subjects[idx]
        subj_data = data['per_subject'][subj]
        if 'confusion_matrix' not in subj_data or subj_data['confusion_matrix'] is None:
            continue
        cm = np.array(subj_data['confusion_matrix'])
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'Subject {subj}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')

    for idx in range(n_plot, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"Confusion Matrices - {data.get('model_type', 'model')}", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved to: {save_path}")


def statistical_test(results_a, results_b, model_a_name='Model A', model_b_name='Model B'):
    """Paired t-test between two models across subjects.

    Args:
        results_a, results_b: Dict[subject, accuracy]

    Returns:
        t-statistic, p-value
    """
    common_subjects = sorted(set(results_a.keys()) & set(results_b.keys()))
    accs_a = [results_a[s] for s in common_subjects]
    accs_b = [results_b[s] for s in common_subjects]

    t_stat, p_val = stats.ttest_rel(accs_a, accs_b)

    print(f"\nPaired t-test: {model_a_name} vs {model_b_name}")
    print(f"  {model_a_name}: {np.mean(accs_a):.4f} +/- {np.std(accs_a):.4f}")
    print(f"  {model_b_name}: {np.mean(accs_b):.4f} +/- {np.std(accs_b):.4f}")
    print(f"  t = {t_stat:.4f}, p = {p_val:.4f}")
    if p_val < 0.05:
        better = model_a_name if np.mean(accs_a) > np.mean(accs_b) else model_b_name
        print(f"  => Significant difference (p < 0.05). {better} is better.")
    else:
        print(f"  => No significant difference (p >= 0.05).")

    return t_stat, p_val


# ─── Full Evaluation Pipeline ────────────────────────────────────────────────

def run_full_evaluation(dataset_name='BNCI2014_001', num_epochs=50):
    """Run complete evaluation: baselines + multi-decoder + comparison.

    1. Train ResEEG baseline (single-decoder)
    2. Train EEG-TCNet baseline (single-decoder)
    3. Load multi-decoder results from checkpoints
    4. Compare all models
    5. Statistical significance tests
    """
    print("=" * 70)
    print("  FULL EVALUATION PIPELINE")
    print(f"  Dataset: {dataset_name}")
    print("=" * 70)

    from EEGNet_residual import ResEEG

    all_results = {}

    # Baseline 1: ResEEG
    print("\n--- ResEEG Baseline ---")
    res_results = train_single_decoder_baseline(
        ResEEG, {'F': 25, 'T': 256}, dataset_name=dataset_name, num_epochs=num_epochs)
    all_results['ResEEG'] = res_results

    # Load multi-decoder results if available
    md_results_path = f'checkpoints/results_hybrid_{dataset_name}.json'
    if os.path.exists(md_results_path):
        with open(md_results_path) as f:
            md_data = json.load(f)
        md_results = {k: v['main_accuracy'] for k, v in md_data['per_subject'].items()}
        # Convert subject keys to match
        first_key = list(res_results.keys())[0]
        if isinstance(first_key, int):
            md_results = {int(k): v for k, v in md_results.items()}
        all_results['MultiDecoder-Hybrid'] = md_results

    md_nott_path = f'checkpoints/results_hybrid_{dataset_name}_no_transformer.json'
    if os.path.exists(md_nott_path):
        with open(md_nott_path) as f:
            nott_data = json.load(f)
        nott_results = {k: v['main_accuracy'] for k, v in nott_data['per_subject'].items()}
        if isinstance(first_key, int):
            nott_results = {int(k): v for k, v in nott_results.items()}
        all_results['MultiDecoder-NoTransformer'] = nott_results

    # Comparison
    if len(all_results) >= 2:
        print("\n--- Model Comparison ---")
        compare_models(all_results,
                       save_path=f'result_imgs/comparison_{dataset_name}.png')

        # Statistical tests
        model_names = list(all_results.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                statistical_test(all_results[model_names[i]],
                                all_results[model_names[j]],
                                model_names[i], model_names[j])

    # Save all results
    os.makedirs('checkpoints', exist_ok=True)
    save_path = f'checkpoints/evaluation_{dataset_name}.json'
    save_data = {name: {str(k): float(v) for k, v in res.items()}
                 for name, res in all_results.items()}
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nEvaluation results saved to: {save_path}")

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate and compare EEG models')
    parser.add_argument('--dataset', type=str, default='BNCI2014_001')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--plot_cm', type=str, default=None,
                        help='Path to results JSON for confusion matrix plotting')
    args = parser.parse_args()

    if args.plot_cm:
        plot_confusion_matrices(args.plot_cm)
    else:
        run_full_evaluation(dataset_name=args.dataset, num_epochs=args.epochs)
