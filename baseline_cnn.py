import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

DATA = Path("/home/rag/datasets/FaceForensicsPP/frames")
DEVICE = torch.device("cpu")
BATCH_SIZE = 16
EPOCHS = 3

with open(DATA / "splits.json") as f:
    splits = json.load(f)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class FrameDataset(Dataset):
    def __init__(self, videos):
        self.samples = []
        for video in videos:
            label = 1 if video["label"] == "fake" else 0
            for frame in video["frames"]:
                if Path(frame).exists():
                    self.samples.append((frame, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return transform(image), torch.tensor(label, dtype=torch.long)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

train_data = FrameDataset(splits["train"])
val_data = FrameDataset(splits["validation"])

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

model = SmallCNN().to(DEVICE)
labels = [label for _, label in train_data.samples]
counts = torch.bincount(torch.tensor(labels))
weights = len(labels) / (2.0 * counts.float())

criterion = nn.CrossEntropyLoss(weight=weights)
print("Class weights:", weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

print("Confusion matrix:")
print(confusion_matrix(all_labels, all_predictions, labels=[0, 1]))

print("Classification report:")
print(classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1],
    target_names=["Real", "Fake"],
    zero_division=0
))

torch.save(model.state_dict(), "baseline_cnn_weighted.pth")
print("Model saved as baseline_cnn.pth")