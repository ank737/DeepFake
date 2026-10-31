import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

ROOT = Path("/home/rag/datasets/FaceForensicsPP")
PREDICTION_ROOT = ROOT / "unet_predictions"
SPLIT_FILE = ROOT / "frames_advo" / "splits_advo.json"

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class UNetPredictionDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return sum(len(item["frames"]) for item in self.samples)

    def __getitem__(self, index):
        for item in self.samples:
            if index < len(item["frames"]):
                original_path = Path(item["frames"][index])
                relative_path = original_path.relative_to(
                    ROOT / "frames_advo"
                )

                prediction_path = (
                    PREDICTION_ROOT / relative_path
                ).with_suffix(".png")

                image = Image.open(prediction_path).convert("L")
                image = transform(image)

                label = 0 if item["label"] == "real" else 1

                return image, torch.tensor(label, dtype=torch.long)

            index -= len(item["frames"])


with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)

train_dataset = UNetPredictionDataset(splits["train"])
val_dataset = UNetPredictionDataset(splits["validation"])
test_dataset = UNetPredictionDataset(splits["test"])

train_loader = DataLoader(
    train_dataset, batch_size=8, shuffle=True
)

val_loader = DataLoader(
    val_dataset, batch_size=8, shuffle=False
)

test_loader = DataLoader(
    test_dataset, batch_size=8, shuffle=False
)

print("========== U-NET GCN DATALOADER TEST ==========")
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Testing samples:", len(test_dataset))

images, labels = next(iter(train_loader))

print("Image shape:", images.shape)
print("Label shape:", labels.shape)
print("DataLoader test completed.")