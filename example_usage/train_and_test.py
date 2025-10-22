import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from moabb import datasets
from moabb.paradigms import MotorImagery
from collections import Counter
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def train_and_save_model(n_classes=5):
    # using BNCI2015_004
    print("loading 5 classes dataset...")
    dataset = datasets.BNCI2015_004()
    paradigm = MotorImagery()
    subjects = dataset.subject_list
    print(f"avaliable subjects: {subjects}")
    print(f"avaliable dataset: {dataset.__class__.__name__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    all_results = []
    subject_performances = {}
    
    # LOO-CV
    for test_subj in subjects:
        print(f"\n{'='*50}")
        print(f"Subject: {test_subj}")
        print(f"{'='*50}")
        train_subs = [s for s in subjects if s != test_subj]
        
        try:
            X_train, y_train, _ = paradigm.get_data(dataset=dataset, subjects=train_subs)
            X_val, y_val, _ = paradigm.get_data(dataset=dataset, subjects=[test_subj])
            
            print(f"train data shape: {X_train.shape}")
            print(f"val data shape: {X_val.shape}")
            print(f"train label distribute: {Counter(y_train)}")
            print(f"val label distribute: {Counter(y_val)}")

            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_val = le.transform(y_val)
            class_names = le.classes_
            print(f"class: {class_names}")
            print(f"class len: {len(class_names)}")
            class_counts = Counter(y_train)
            total_samples = len(y_train)
            weights = torch.tensor([total_samples / (len(class_counts) * class_counts[i]) for i in range(len(class_counts))]).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
            
            print(f"class weight: {weights.cpu().numpy()}")
            
            # normalization
            scaler = StandardScaler()
            X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
            X_train = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
            X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
            X_val = scaler.transform(X_val_reshaped).reshape(X_val.shape)
            
            # reshape
            C, T = X_train.shape[1], X_train.shape[2]  # (C, T)
            n_classes_actual = len(np.unique(y_train))
            
            print(f"channels: {C}, time points: {T}, number of classes: {n_classes_actual}")

            X_train = X_train[:, np.newaxis, :, :]  # (trials, 1, channels, time)
            X_val = X_val[:, np.newaxis, :, :]
            
            X_train = torch.FloatTensor(X_train)
            y_train = torch.LongTensor(y_train)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.LongTensor(y_val)
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
            
            # model initialize
            model = _EEG_TCNet(n_chan=C, n_cls=n_classes_actual).to(device)
            print(f"model size: {sum(p.numel() for p in model.parameters())}")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            num_epochs = 30
            best_val_acc = 0
            best_model_state = None
            
            # train
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                correct = 0
                total = 0
                
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                scheduler.step()
                
                avg_train_loss = train_loss / len(train_loader)
                train_acc = 100 * correct / total
                
                # val
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
                
                # save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Loss: {avg_train_loss:.4f}, "
                          f"Train Acc: {train_acc:.2f}%, "
                          f"Val Acc: {val_acc:.4f}, "
                          f"LR: {scheduler.get_last_lr()[0]:.6f}")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            model.eval()
            final_preds, final_true = [], []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    _, preds = torch.max(outputs, 1)
                    final_preds.extend(preds.cpu().numpy())
                    final_true.extend(batch_y.cpu().numpy())
            
            final_val_acc = accuracy_score(final_true, final_preds)
            cm = confusion_matrix(final_true, final_preds)
            all_results.append(final_val_acc)
            subject_performances[test_subj] = {
                'accuracy': final_val_acc,
                'confusion_matrix': cm,
                'class_names': class_names,
                'n_classes': n_classes_actual
            }
            
            print(f"\nSubject: {test_subj} final results:")
            print(f"best val_acc: {best_val_acc:.4f}")
            print(classification_report(final_true, final_preds, target_names=class_names))
            
        except Exception as e:
            print(f"exception occured when processing subject {test_subj}: {e}")
            continue

    print(f"\n{'='*60}")
    print("Total Information")
    print(f"{'='*60}")
    
    if all_results:
        mean_acc = np.mean(all_results)
        std_acc = np.std(all_results)
        
        print("\nSubjects:")
        for subj, perf in subject_performances.items():
            print(f"Subject {subj}: {perf['accuracy']:.4f}")
        
        print(f"\navg acc: {mean_acc:.4f} ± {std_acc:.4f}")
        
        print("\nusing all data to train...")
        X_all, y_all, _ = paradigm.get_data(dataset=dataset, subjects=subjects)
        
        le_final = LabelEncoder()
        y_all = le_final.fit_transform(y_all)
        class_names_final = le_final.classes_
        print(f"final class name: {class_names_final}")
        
        scaler_final = StandardScaler()
        X_all_reshaped = X_all.reshape(X_all.shape[0], -1)
        X_all = scaler_final.fit_transform(X_all_reshaped).reshape(X_all.shape)
        
        C_final, T_final = X_all.shape[1], X_all.shape[2]
        n_classes_final = len(np.unique(y_all))
        print(f"final channels: {C_final}, final time points: {T_final}, classes: {n_classes_final}")
        X_all = X_all[:, np.newaxis, :, :]
        X_all = torch.FloatTensor(X_all)
        y_all = torch.LongTensor(y_all)
        full_loader = DataLoader(TensorDataset(X_all, y_all), batch_size=64, shuffle=True)
        final_model = _EEG_TCNet(n_chan=C_final, n_cls=n_classes_final).to(device)
        optimizer_final = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler_final = torch.optim.lr_scheduler.StepLR(optimizer_final, step_size=10, gamma=0.5)
        class_counts_final = Counter(y_all.numpy())
        weights_final = torch.tensor([len(y_all) / (n_classes_final * class_counts_final[i]) for i in range(n_classes_final)]).to(device)
        criterion_final = nn.CrossEntropyLoss(weight=weights_final)
        num_epochs_final = 30
        best_final_acc = 0
        best_final_state = None
        split_idx = int(0.8 * len(X_all))
        X_train_final, X_val_final = X_all[:split_idx], X_all[split_idx:]
        y_train_final, y_val_final = y_all[:split_idx], y_all[split_idx:]
        
        train_loader_final = DataLoader(TensorDataset(X_train_final, y_train_final), batch_size=64, shuffle=True)
        val_loader_final = DataLoader(TensorDataset(X_val_final, y_val_final), batch_size=64, shuffle=False)
        
        for epoch in range(num_epochs_final):
            final_model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_x, batch_y in train_loader_final:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer_final.zero_grad()
                outputs = final_model(batch_x)
                loss = criterion_final(outputs, batch_y)
                loss.backward()
                optimizer_final.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            scheduler_final.step()
            
            avg_train_loss = train_loss / len(train_loader_final)
            train_acc = 100 * correct / total

            final_model.eval()
            val_preds_final, val_true_final = [], []
            with torch.no_grad():
                for batch_x, batch_y in val_loader_final:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = final_model(batch_x)
                    _, preds = torch.max(outputs, 1)
                    val_preds_final.extend(preds.cpu().numpy())
                    val_true_final.extend(batch_y.cpu().numpy())
            
            val_acc_final = accuracy_score(val_true_final, val_preds_final)
            
            if val_acc_final > best_final_acc:
                best_final_acc = val_acc_final
                best_final_state = final_model.state_dict().copy()
            
            if (epoch + 1) % 5 == 0:
                print(f"Final Epoch {epoch+1}/{num_epochs_final} - "
                      f"Loss: {avg_train_loss:.4f}, "
                      f"Train Acc: {train_acc:.2f}%, "
                      f"Val Acc: {val_acc_final:.4f}")

        if best_final_state is not None:
            final_model.load_state_dict(best_final_state)

        scaler_params = {
            'mean': scaler_final.mean_,
            'var': scaler_final.var_,
            'scale': scaler_final.scale_,
            'n_samples_seen': scaler_final.n_samples_seen_
        }
        checkpoint = {
            'model_state_dict': final_model.state_dict(),
            'scaler_params': scaler_params,
            'n_channels': C_final,
            'n_times': T_final,
            'n_classes': n_classes_final,
            'class_names': class_names_final.tolist() if hasattr(class_names_final, 'tolist') else class_names_final
        }
        model_path = 'trained_model_weights.pth'
        torch.save(checkpoint, model_path)
        print(f"\nmodel saved as: {model_path}")
        print(f"final acc: {best_final_acc:.4f}")
        
        return subject_performances, mean_acc, std_acc, model_path
    else:
        print("no results")
        return {}, 0, 0, None

if __name__ == "__main__":
    print("start training...")
    results, mean_acc, std_acc, saved_path = train_and_save_model()
    
    if saved_path:
        print(f"\nfinish! model saved as {saved_path}")
        print(f"CV val acc: {mean_acc:.4f} ± {std_acc:.4f}")
    else:
        print("fail!")
