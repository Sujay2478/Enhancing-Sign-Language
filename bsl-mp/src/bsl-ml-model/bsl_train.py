import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import traceback
import torch.onnx

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "data/one_hand_dataset.csv"
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH, header=None)
df.columns = [f"f{i}" for i in range(df.shape[1] - 1)] + ["label"]

print("Data shape:", df.shape)
print("Columns:", df.columns.tolist()[:5], "...", df.columns.tolist()[-5:])

# Separate features and labels
X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
print(f"Detected {num_classes} unique signs.")

# ============================================================
# DATASET
# ============================================================
class BSLSignedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = BSLSignedDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# MODEL
# ============================================================
class BSLNet(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_classes=26):
        super(BSLNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = BSLNet(input_dim=X.shape[1], num_classes=num_classes).to(DEVICE)
print(model)

# ============================================================
# TRAINING
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
        preds = outputs.argmax(1)
        correct += (preds == yb).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            outputs = model(Xb)
            loss = criterion(outputs, yb)
            total_loss += loss.item() * Xb.size(0)
            preds = outputs.argmax(1)
            correct += (preds == yb).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

# ============================================================
# SAVE PYTORCH MODEL
# ============================================================
os.makedirs("models", exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "encoder_classes": encoder.classes_
}, "models/bsl_sign_model.pth")
print("✅ Model saved to models/bsl_sign_model.pth")

# ============================================================
# EXPORT TO ONNX (browser-safe)
# ============================================================
try:
    print("🔧 Starting ONNX export (external path for browser)...")

    import onnx
    from torch.autograd import Variable

    model_cpu = model.to("cpu").eval()
    dummy_input = Variable(torch.randn(1, X.shape[1], dtype=torch.float32))

    onnx_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "../../public/models/bsl_sign_model.onnx"
    ))
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    if os.path.exists(onnx_path + ".data"):
        os.remove(onnx_path + ".data")

    # ✅ Force export through the standard torch.onnx interface but save IR9
    torch.onnx.export(
        model_cpu,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,            # most web-compatible opset
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None
    )

    # ✅ Downgrade IR version to v9 for onnxruntime-web
    model_proto = onnx.load(onnx_path)
    model_proto.ir_version = 9
    onnx.save(model_proto, onnx_path)

    file_size = os.path.getsize(onnx_path)
    print(f"✅ Browser-safe ONNX export succeeded: {onnx_path}")
    print(f"📦 File size: {file_size / 1024:.2f} KB")
    print("✅ Ready for onnxruntime-web!")

except Exception:
    print("❌ ONNX export failed:")
    traceback.print_exc()
