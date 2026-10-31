import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


SPLIT_FILE = Path(
    "/home/rag/datasets/FaceForensicsPP/frames_advo/splits_advo.json"
)

BATCH_SIZE = 8
IMAGE_SIZE = 224


class DeepfakeDataset(Dataset):

    def __init__(self, split):

        with open(SPLIT_FILE, "r") as f:
            data = json.load(f)

        self.samples = []

        for item in data[split]:

            label = 0 if item["label"] == "real" else 1

            for frame in item["frames"]:
                image_path = Path(frame)

                if image_path.exists():
                    self.samples.append((image_path, label))

        self.transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("L")
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# Datasets
train_dataset = DeepfakeDataset("train")
val_dataset = DeepfakeDataset("validation")
test_dataset = DeepfakeDataset("test")


# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# Test
images, labels = next(iter(train_loader))

print("========== GCN DATALOADER TEST ==========")
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Testing samples:", len(test_dataset))
print("Image shape:", images.shape)
print("Label shape:", labels.shape)
print("Labels:", labels)
print("DataLoader test completed.")