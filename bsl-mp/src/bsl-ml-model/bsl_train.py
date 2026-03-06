import os
import json
import traceback
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder


# ============================================================
# CONFIGURATION
# ============================================================
ONE_HAND_CSV = os.path.join("data", "one_hand_dataset.csv")     # 63 + label
TWO_HAND_CSV = os.path.join("data", "two_hand_dataset.csv")     # 126 + label

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
SEED = 42

INPUT_DIM = 126

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ============================================================
# REPRO
# ============================================================
def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ============================================================
# LOADING HELPERS
# ============================================================
def load_csv_features_labels(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    CSV has no header.
    Last col is label, preceding cols are float features.
    """
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(str)
    return X, y


def pad_to_dim(X: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Pads feature vectors with zeros on the right to match target_dim.
    """
    if X.shape[1] == target_dim:
        return X
    if X.shape[1] > target_dim:
        raise ValueError(f"Feature dim {X.shape[1]} > target dim {target_dim}.")
    pad_width = target_dim - X.shape[1]
    return np.pad(X, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)


# ============================================================
# NORMALIZATION (train-split stats)
# ============================================================
@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray

    def to_jsonable(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def compute_norm_stats(X: np.ndarray, eps: float = 1e-6) -> NormStats:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < eps, 1.0, std)  # avoid div by tiny numbers
    return NormStats(mean=mean, std=std)


def apply_norm(X: np.ndarray, stats: NormStats) -> np.ndarray:
    return (X - stats.mean) / stats.std


# ============================================================
# DATA AUGMENTATION
# ============================================================
def augment_features(
    x: np.ndarray,
    noise_std: float = 0.015,
    scale_range: Tuple[float, float] = (0.95, 1.05),
    drop_prob: float = 0.0,
) -> np.ndarray:
    """
    x is (126,). Assumes x is already roughly normalized/centered per sample (your pipeline does).
    We do small jitter + small global scale.
    Optional feature-drop for robustness (usually keep low or 0).
    """
    out = x.copy()

    s = np.random.uniform(scale_range[0], scale_range[1])
    out *= s

    out += np.random.normal(0.0, noise_std, size=out.shape).astype(np.float32)

    if drop_prob > 0:
        mask = (np.random.rand(out.shape[0]) > drop_prob).astype(np.float32)
        out *= mask

    return out


# ============================================================
# DATASET
# ============================================================
class BSLDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        norm: Optional[NormStats] = None,
        augment: bool = False,
    ):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.norm = norm
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]

        if self.norm is not None:
            x = (x - self.norm.mean) / self.norm.std

        if self.augment:
            x = augment_features(
                x,
                noise_std=0.015,
                scale_range=(0.95, 1.05),
                drop_prob=0.0,
            )

        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


# ============================================================
# MODEL
# ============================================================
class BSLNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# TRAIN / EVAL
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        n += xb.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        n += xb.size(0)

    return total_loss / n, correct / n


# ============================================================
# MAIN
# ============================================================
def main():
    # ---------- load both datasets ----------
    X1, y1 = load_csv_features_labels(ONE_HAND_CSV)
    X2, y2 = load_csv_features_labels(TWO_HAND_CSV)

    print(f"one_hand: X={X1.shape}, y={y1.shape}")
    print(f"two_hand: X={X2.shape}, y={y2.shape}")

    # ---------- pad one-hand to 126 ----------
    X1 = pad_to_dim(X1, INPUT_DIM)
    X2 = pad_to_dim(X2, INPUT_DIM)

    # ---------- combine ----------
    X_all = np.concatenate([X1, X2], axis=0)
    y_all = np.concatenate([y1, y2], axis=0)

    # ---------- encode labels across BOTH datasets ----------
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_all)
    num_classes = len(encoder.classes_)
    print(f"Detected {num_classes} unique signs (combined).")

    # ---------- train/val split ----------
    full_idx = np.arange(len(X_all))
    full_ds_for_split = list(full_idx)

    train_size = int(0.8 * len(full_ds_for_split))
    val_size = len(full_ds_for_split) - train_size

    train_idx, val_idx = random_split(full_ds_for_split, [train_size, val_size])

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)

    X_train_raw = X_all[train_idx]
    y_train = y_enc[train_idx]
    X_val_raw = X_all[val_idx]
    y_val = y_enc[val_idx]

    # ---------- compute normalization on TRAIN ONLY ----------
    norm = compute_norm_stats(X_train_raw)
    print("Computed train normalization stats.")

    # ---------- datasets/loaders ----------
    train_ds = BSLDataset(X_train_raw, y_train, norm=norm, augment=True)
    val_ds = BSLDataset(X_val_raw, y_val, norm=norm, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---------- model ----------
    model = BSLNet(input_dim=INPUT_DIM, hidden_dim=256, num_classes=num_classes).to(DEVICE)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ---------- train loop ----------
    best_val = 0.0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if val_acc > best_val:
            best_val = val_acc

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} "
            f"| Best Val: {best_val:.4f}"
        )

    # ---------- save pytorch checkpoint ----------
    os.makedirs("models", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_classes": encoder.classes_,
            "input_dim": INPUT_DIM,
            "norm_mean": norm.mean.astype(np.float32),
            "norm_std": norm.std.astype(np.float32),
        },
        os.path.join("models", "bsl_sign_model.pth"),
    )
    print("Model saved to models/bsl_sign_model.pth")

    # ---------- save labels json for frontend ----------
    labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "bsl_labels.json"))
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(encoder.classes_.tolist(), f, ensure_ascii=False, indent=2)
    print(f"Label map saved to {labels_path}")

    # ---------- save normalization json for frontend (optional but recommended) ----------
    norm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "bsl_norm.json"))
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(norm.to_jsonable(), f, ensure_ascii=False, indent=2)
    print(f"Normalization stats saved to {norm_path}")

    # ---------- export to ONNX (browser-safe) ----------
    try:
        print("🔧 Starting ONNX export (browser-safe)...")

        import onnx
        from torch.autograd import Variable

        model_cpu = model.to("cpu").eval()
        dummy_input = Variable(torch.randn(1, INPUT_DIM, dtype=torch.float32))

        onnx_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../public/models/bsl_sign_model.onnx")
        )
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

        if os.path.exists(onnx_path + ".data"):
            os.remove(onnx_path + ".data")

        torch.onnx.export(
            model_cpu,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,
            verbose=False,
        )

        model_proto = onnx.load(onnx_path)
        model_proto.ir_version = 9
        onnx.save(model_proto, onnx_path)

        file_size = os.path.getsize(onnx_path)
        print(f"ONNX export succeeded: {onnx_path}")
        print(f"File size: {file_size / 1024:.2f} KB")
        print("Ready for onnxruntime-web!")

    except Exception:
        print("ONNX export failed:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
